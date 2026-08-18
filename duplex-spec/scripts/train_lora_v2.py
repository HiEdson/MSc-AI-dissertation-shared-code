"""v2 training: LoRA on the frozen q8 Moshi final block, trained JOINTLY with the head.

No feature cache: the q8 backbone runs a LIVE forward every batch; adapters + head are
updated through it.

Design notes (learned the hard way):
  * single LMGen + per-conversation `with lm_gen.streaming(1)` --- avoids "already
    streaming!" state leaks;
  * RANDOM training window per conversation per epoch. Training on frame 0..200 of every
    call taught the head to predict silence and collapsed it (99% rollback at eval);
  * PREFILL: a random window opens a fresh, EMPTY KV cache, so frame `start` looks like
    the beginning of a conversation. v0's cached features always had full history. We
    therefore stream `--prefill` frames before the window under no_grad, without loss, to
    warm the cache so features match v0's distribution;
  * a real VALIDATION split from the TRAINING pool (never the held-out eval set), with a
    FIXED window so the val signal is comparable across epochs;
  * best-checkpoint on val loss; --init-head warm-start from the trained v0 head;
  * checkpoint saved in the dict format eval_speculative.py expects.

Sanity check on the first run: with --init-head and prefill, epoch 0 val should start
near the v0 val loss (~4.0), because LoRA is an exact identity at init (B=0). If it starts
much higher, the features still do not match v0's distribution --- stop and investigate
rather than training on.

PREREQ: scripts/probe_qlinear_grad.py must print "VERDICT: YES".

Typical run:
    python scripts/train_lora_v2.py --pairs-dir tokens/ --device cuda \
      --target-modules "self_attn.in_projs.0,self_attn.out_projs.0" \
      --max-frames 2000 --epochs 10 --init-head head_v0.pt \
      --out head_lora_v2.pt --adapters-out lora_v2.pt

HONEST STATUS: unvalidated in the assistant sandbox (no GPU / no Moshi).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from moshi.utils.compile import no_cuda_graph

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

def resolve(obj, dotted):
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, required=True,
                    help="Stage-A token .npy files (TRAINING pool); features are live")
    ap.add_argument("--out", type=Path, default=Path("head_lora_v2.pt"))
    ap.add_argument("--adapters-out", type=Path, default=Path("lora_v2.pt"))
    ap.add_argument("--init-head", type=Path, default=None,
                    help="warm-start the head from a trained v0 checkpoint (recommended)")
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers-attr", default="transformer.layers")
    ap.add_argument("--target-modules", default=None,
                    help="comma-separated final-block linear names from the probe")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=32,
                    help="grad accumulation; effective batch = batch*accum (v0 used 512)")
    ap.add_argument("--max-frames", type=int, default=2000,
                    help="frames per conversation window")
    ap.add_argument("--prefill", type=int, default=200,
                    help="frames streamed before the window (no loss) to warm the KV cache")
    ap.add_argument("--val-convs", type=int, default=4,
                    help="conversations held out of --pairs-dir for validation")
    ap.add_argument("--val-max-frames", type=int, default=None,
                    help="val window length (default: same as --max-frames)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5, help="LoRA adapter lr")
    ap.add_argument("--head-lr", type=float, default=1e-5,
                    help="head lr (low: the head is warm-started and the effective batch "
                         "is far smaller than v0's 512)")
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch
        from moshi.models import LMGen, loaders
        try:
            from duplex_spec_v2.lora import inject_lora, lora_parameters, save_adapters
        except ImportError:
            from duplex_spec.lora import inject_lora, lora_parameters, save_adapters
        from duplex_spec.head import MultiStepTPPHead, tpp_loss
    except ImportError as e:
        sys.exit(f"Import failed ({e}). Ensure moshi + duplex_spec(.lora/.head) import.")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    dev, dt = args.device, getattr(torch, args.dtype)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ---- frozen q8 backbone ----
    ckpt = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    lm = ckpt.get_moshi(device=dev, dtype=dt)
    try:
        from moshi.utils.quantize import QLinear
        for m in lm.modules():
            if isinstance(m, QLinear):
                m.weight_scb.data = m.weight_scb.data.float()
    except Exception:
        pass
    for p in lm.parameters():
        p.requires_grad_(False)

    final = resolve(lm, args.layers_attr)[-1]
    if args.target_modules:
        targets = [s.strip() for s in args.target_modules.split(",")]
    else:
        from torch import nn
        targets = [n for n, m in final.named_modules() if isinstance(m, nn.Linear)]
    adapters = inject_lora(final, targets, r=args.lora_r, alpha=args.lora_alpha, dtype=dt)
    if len(adapters) == 0:
        sys.exit("No modules wrapped --- pass --target-modules from the probe listing.")
    n_ad = sum(p.numel() for p in lora_parameters(adapters))
    print(f"[lora] trainable {n_ad/1e6:.2f}M adapter params on {len(adapters)} layers")

    # ---- head ----
    H = 4096
    head = MultiStepTPPHead(hidden_dim=H, n_channels=2, n_codebooks=8,
                            codebook_size=2048, horizon=args.horizon).to(dev)
    if args.init_head:
        ck = torch.load(args.init_head, map_location="cpu")
        head.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
        print(f"[head] warm-started from {args.init_head}")
    else:
        print("[head] WARNING: random init. Warm-starting from head_v0.pt is strongly "
              "recommended.")

    params = [{"params": list(lora_parameters(adapters)), "lr": args.lr},
              {"params": list(head.parameters()), "lr": args.head_lr}]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params); print("[opt] bitsandbytes AdamW8bit")
    except Exception:
        opt = torch.optim.AdamW(params); print("[opt] torch AdamW")
    print(f"[opt] lr(lora)={args.lr:g} lr(head)={args.head_lr:g} "
          f"effective batch={args.batch*args.accum}")

    # ---- train / val split (val from the TRAINING pool, never the eval set) ----
    files = sorted(args.pairs_dir.glob("*.npy"))
    if not files:
        sys.exit(f"No .npy token files in {args.pairs_dir}")
    perm = rng.permutation(len(files))
    n_val = min(args.val_convs, max(1, len(files) // 10))
    val_files = [files[i] for i in perm[:n_val]]
    train_files = [files[i] for i in perm[n_val:]]
    val_W = args.val_max_frames or args.max_frames
    print(f"[data] {len(train_files)} train / {len(val_files)} val conversations "
          f"(LIVE features); train window={args.max_frames} (random offset), "
          f"val window={val_W} (fixed), prefill={args.prefill}")

    needed = lm.num_codebooks - lm.dep_q - 1
    K = args.horizon
    
    _raw = getattr(LMGen._step, "__wrapped__", None)
    if _raw is None:
        sys.exit("Could not unwrap LMGen._step (no __wrapped__); cannot train through it.")
    def _step_grad(gen, *a, **kw):
        return _raw(gen, *a, **kw)          # undecorated -> grad flows
    print("[grad] using undecorated LMGen._step (bypasses @torch.no_grad)")
    
    lm_gen = LMGen(lm)
    checked = {"done": False}

    def frame_tensor(tokens, ch, t):
        return torch.from_numpy(tokens[ch, :, t]).to(dev).long()[None, :, None]

    def window(tokens, W, random_offset):
        """Return (window_including_prefill, n_prefill_frames)."""
        T_full = tokens.shape[2]
        if T_full <= W:
            return tokens, 0
        start = int(rng.integers(0, T_full - W)) if random_offset else (T_full - W) // 2
        pre = min(args.prefill, start)
        return tokens[:, :, start - pre:start + W], pre


    def _deep_detach(o, seen=None):
        if seen is None:
            seen = set()
        if id(o) in seen:
            return o
        seen.add(id(o))
        if torch.is_tensor(o):
            return o.detach() if o.grad_fn is not None else o
        if isinstance(o, dict):
            for k, v in list(o.items()):
                o[k] = _deep_detach(v, seen)
            return o
        if isinstance(o, list):
            for i, v in enumerate(o):
                o[i] = _deep_detach(v, seen)
            return o
        if isinstance(o, tuple):
            return o
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            for k, v in list(d.items()):
                try:
                    setattr(o, k, _deep_detach(v, seen))
                except Exception:
                    pass
        return o

    def detach_streaming(module):
        st = getattr(module, "_streaming_state", None)
        if st is not None:
            _deep_detach(st)
        for c in module.children():
            detach_streaming(c)
            
    def run_conv(tokens, train: bool, n_prefill: int = 0, tbptt_chunk: int = 32):
        """One conversation window: live forward + loss.
        Uses chunked Truncated Backpropagation Through Time (TBPTT) to 
        preserve KV cache gradients.
        """
        T = tokens.shape[2]
        feats, tgts = [], []
        tot, n, local_step = 0.0, 0, 0
        
        with lm_gen.streaming(1):
            for t in range(T):
                user = frame_tensor(tokens, 0, t)[:, :needed]
                moshi = frame_tensor(tokens, 1, t)[:, : lm.dep_q]
                
                # 1. Warm cache without gradients (Prefill)
                if t < n_prefill:
                    with torch.no_grad():
                        lm_gen._step(user, depformer_replace_tokens=moshi)
                    continue
                
                # 2. Live forward pass (adds to the computational graph)
                with no_cuda_graph():
                    out = _step_grad(lm_gen, user, depformer_replace_tokens=moshi)
                
                if out is None:
                    continue
                    
                _tok, hstate = out
                h = hstate[:, 0] if hstate.dim() == 3 else hstate
                
                if t + K < T:
                    feats.append(h)
                    tgts.append(tokens[:, :, t + 1:t + 1 + K])
                
                # 3. TBPTT Chunk Processing
                # Trigger backward pass only when the chunk is full
                if len(feats) >= tbptt_chunk or t == T - 1:
                    if not feats:
                        continue
                    
                    # hb shape: [chunk_size, H]
                    hb = torch.cat(feats, 0).float()
                    yb = torch.from_numpy(np.stack(tgts)).to(dev).long()
                    
                    # tpp_loss computes the mean over all items, effectively averaging the chunk
                    loss = tpp_loss(head(hb), yb)
                    
                    if torch.isfinite(loss):
                        if train:
                            # Scale by accum to maintain effective learning rate
                            (loss / args.accum).backward()
                            
                            # SEVER THE GRAPH THROUGH TIME
                            # This preserves gradients within the chunk, but prevents 
                            # PyTorch from unrolling the entire 2000-frame window.
                            detach_streaming(lm)
                            
                            if not checked["done"]:
                                g = adapters[0].B.grad
                                print(f"[chk] hb.requires_grad={hb.requires_grad} "
                                      f"| A.grad={'None' if g is None else f'{g.norm().item():.3e}'}")
                                checked["done"] = True
                                
                    tot += loss.item() * len(feats)
                    n += len(feats)
                    
                    # Reset chunk buffers
                    feats, tgts = [], []
                    local_step += 1
                    
                    # 4. Optimizer Step
                    if train and local_step % args.accum == 0:
                        if args.clip_grad > 0:
                            torch.nn.utils.clip_grad_norm_(
                                list(lora_parameters(adapters)) + list(head.parameters()),
                                args.clip_grad)
                        opt.step()
                        opt.zero_grad()
                        
        return tot, n

    # ---- epochs ----
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_val = float("inf")
    for ep in range(args.epochs):
        head.train()
        ttot, tn = 0.0, 0
        for f in train_files:
            tk, pre = window(np.load(f), args.max_frames, random_offset=True)
            a, b = run_conv(tk, train=True, n_prefill=pre)
            ttot += a; tn += b
        opt.zero_grad()

        head.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for f in val_files:
                tk, pre = window(np.load(f), val_W, random_offset=False)   # FIXED window
                a, b = run_conv(tk, train=False, n_prefill=pre)
                vtot += a; vn += b

        tr = ttot / max(tn, 1); vl = vtot / max(vn, 1)
        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        best = ""
        if vl < best_val:
            best_val = vl; best = "  *best*"
            save_adapters(adapters, args.adapters_out)
            torch.save({"state_dict": head.state_dict(), "head_type": "independent",
                        "horizon": K, "hidden_dim": H, "n_channels": 2,
                        "n_codebooks": 8, "epoch": ep, "val_loss": float(vl)}, args.out)
        print(f"[ep {ep}] train={tr:.3f} val={vl:.3f} | peak={peak:.2f} GB{best}")

    print(f"[save] best val_loss={best_val:.3f} -> adapters {args.adapters_out}, "
          f"head {args.out}")
    print("Eval: re-load q8, re-inject LoRA with the SAME --target-modules, load these "
          "adapters, extract features live, then run eval_speculative.py.")


if __name__ == "__main__":
    main()
