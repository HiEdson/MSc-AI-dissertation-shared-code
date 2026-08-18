"""v2 training: LoRA on the frozen q8 Moshi final block, trained JOINTLY with the head.

This is the fundamentally different training regime from v0/v1: there is NO feature
cache. The q8 backbone runs a LIVE forward on every batch, adapters + head are updated
through it. Memory is the binding constraint on 16 GB, so this script leans on:
  * final-block-only LoRA (few trainable params);
  * gradient checkpointing on the backbone (recompute activations in backward);
  * bf16 compute, batch size 1 with gradient accumulation;
  * an 8-bit optimiser if bitsandbytes is available (else AdamW).

PREREQUISITES (do these first, in order):
  1. Run scripts/probe_qlinear_grad.py --- it MUST print "VERDICT: YES". If not, this
     script cannot train and the approach needs redesign (adapters between blocks).
  2. Use the final-block linear names the probe printed as --target-modules.

HONEST STATUS: unvalidated in the assistant sandbox (no GPU / no Moshi, and this needs
a live ~q8-Moshi backward that likely sits near the 16 GB limit). Treat the first run as
a memory/feasibility test: start with --max-frames small, watch peak memory, and only
scale up if it fits. If it OOMs, the fallbacks (below) are your levers.

FALLBACKS IF IT OOMS (in order of preference):
  --grad-checkpoint (already default on)  |  --max-frames smaller (shorter windows)
  --accum larger with --batch 1           |  --lora-r smaller (4 or 2)
  wrap fewer target modules (attn only, not MLP)
If it still OOMs with all of these, the live q8 backward does not fit 16 GB and Path A
needs a rented GPU --- report the peak-memory number and we rethink.
"""
from __future__ import annotations
import argparse, sys, math
from pathlib import Path
import numpy as np


def resolve(obj, dotted):
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, required=True,
                    help="Stage-A token .npy files (targets). Features are computed LIVE, "
                         "so only tokens are needed here --- no cached .npz.")
    ap.add_argument("--out", type=Path, default=Path("head_lora_v2.pt"))
    ap.add_argument("--adapters-out", type=Path, default=Path("lora_v2.pt"))
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers-attr", default="transformer.layers")
    ap.add_argument("--target-modules", default=None,
                    help="comma-separated final-block linear names from the probe "
                         "(e.g. 'self_attn.out_proj,gating.linear_in'). "
                         "If omitted, wraps all linears in the final block.")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--max-frames", type=int, default=500,
                    help="frames per conversation window (keep small first to fit memory)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    args = ap.parse_args()

    try:
        import torch
        from moshi.models import LMGen, loaders
        #sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from duplex_spec.lora import inject_lora, lora_parameters, save_adapters
        # reuse the v0 head + loss from the original project (adjust path as needed)
        from duplex_spec.head import MultiStepTPPHead, tpp_loss
    except ImportError as e:
        sys.exit(f"Import failed ({e}). Ensure moshi, the v2 src, and the original "
                 f"duplex_spec.head are on PYTHONPATH.")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    dev, dt = args.device, getattr(torch, args.dtype)

    # ---- load frozen q8 backbone ----
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
        p.requires_grad_(False)                     # freeze everything first

    layers = resolve(lm, args.layers_attr)
    final = layers[-1]

    # ---- inject LoRA into the final block ----
    if args.target_modules:
        targets = [s.strip() for s in args.target_modules.split(",")]
    else:
        from torch import nn
        targets = [n for n, m in final.named_modules() if isinstance(m, nn.Linear)]
        try:
            from moshi.utils.quantize import QLinear as _QL
            targets += [n for n, m in final.named_modules() if isinstance(m, _QL)]
        except Exception:
            pass
    adapters = inject_lora(final, targets, r=args.lora_r, alpha=args.lora_alpha, dtype=dt)
    if len(adapters) == 0:
        sys.exit("No modules wrapped --- pass --target-modules from the probe listing.")
    # base stays frozen (we froze lm above); only adapter A/B are trainable
    n_train = sum(p.numel() for p in lora_parameters(adapters))
    print(f"[lora] trainable {n_train/1e6:.2f}M adapter params on {len(adapters)} layers")

    if args.grad_checkpoint:
        # best-effort: enable checkpointing if the transformer supports it
        for attr in ("gradient_checkpointing_enable", "set_grad_checkpointing"):
            fn = getattr(lm, attr, None) or getattr(resolve(lm, "transformer"), attr, None)
            if callable(fn):
                try:
                    fn(True); print(f"[mem] gradient checkpointing via {attr}")
                    break
                except Exception:
                    pass
        else:
            print("[mem] WARNING: could not enable gradient checkpointing automatically; "
                  "if it OOMs, this is the first thing to wire up manually.")

    # ---- head (v0 independent) ----
    H = 4096
    head = MultiStepTPPHead(hidden_dim=H, n_channels=2, n_codebooks=8,
                            codebook_size=2048, horizon=args.horizon).to(dev)

    # ---- optimiser: 8-bit if available, else AdamW ----
    params = [{"params": list(lora_parameters(adapters)), "lr": args.lr},
              {"params": list(head.parameters()), "lr": args.head_lr}]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params)
        print("[opt] using bitsandbytes AdamW8bit (memory-light)")
    except Exception:
        opt = torch.optim.AdamW(params)
        print("[opt] using torch AdamW (bitsandbytes not found)")

    # ---- data: token files only (features computed live) ----
    files = sorted(args.pairs_dir.glob("*.npy"))
    if not files:
        sys.exit(f"No .npy token files in {args.pairs_dir}")
    print(f"[data] {len(files)} conversations; features computed LIVE (no cache)")

    needed = lm.num_codebooks - lm.dep_q - 1
    K = args.horizon

    def frame_tensor(tokens, ch, t):
        return torch.from_numpy(tokens[ch, :, t]).to(dev).long()[None, :, None]

    # ---- training loop ----
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step = 0
    lm_gen = LMGen(lm)
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for f in files:
            tokens = np.load(f)                       # [2, K, T]
            #T = min(tokens.shape[2], args.max_frames)
            T_full = tokens.shape[2]
            start = np.random.randint(0, max(1, T_full - args.max_frames))
            tokens = tokens[:, :, start:start + args.max_frames]
            T = tokens.shape[2]
            feats, tgts = [], []
            # LIVE forward through the (now LoRA-adapted) backbone, teacher-forced
            with lm_gen.streaming(1):
                for t in range(T):
                    user = frame_tensor(tokens, 0, t)[:, :needed]
                    moshi = frame_tensor(tokens, 1, t)[:, : lm.dep_q]
                    out = lm_gen._step(user, depformer_replace_tokens=moshi)
                    if out is None:
                        continue
                    _tok, hstate = out
                    h = hstate[:, 0] if hstate.dim() == 3 else hstate  # [1, H], grad-enabled
                    # collect targets for horizons k=1..K that exist
                    if t + K < T:
                        feats.append(h)
                        tgts.append(tokens[:, :, t + 1:t + 1 + K])     # [2, K, K] targets
                    # train in small chunks to bound activation memory
                    if len(feats) >= args.batch or t == T - 1:
                        if not feats:
                            continue
                        hb = torch.cat(feats, 0).float()
                        yb = torch.from_numpy(np.stack(tgts)).to(dev).long()  # [B,2,K,K]
                        loss = tpp_loss(head(hb), yb) / args.accum
                        if torch.isfinite(loss):
                            loss.backward()
                            tot += loss.item() * args.accum * len(feats); n += len(feats)
                        feats, tgts = [], []
                        step += 1
                        if step % args.accum == 0:
                            if args.clip_grad > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    list(lora_parameters(adapters)) + list(head.parameters()),
                                    args.clip_grad)
                            opt.step(); opt.zero_grad()
        peak = torch.cuda.max_memory_allocated()/1e9 if dev == "cuda" else 0
        print(f"[ep {ep}] train={tot/max(n,1):.3f}  peak={peak:.2f} GB")
        
        
    # ---- save adapters + head (NOT the whole backbone) ----
    save_adapters(adapters, args.adapters_out)
    torch.save({"state_dict": head.state_dict(), "head_type": "independent",
            "horizon": args.horizon, "hidden_dim": 4096,
            "n_channels": 2, "n_codebooks": 8, "val_loss": "v2-lora"}, args.out)
    #torch.save(head.state_dict(), args.out)
    print(f"[save] adapters -> {args.adapters_out}, head -> {args.out}")
    print("Reminder: at eval time you must re-load the q8 backbone, re-inject LoRA with "
          "the SAME target modules, load these adapter weights, then run features live.")


if __name__ == "__main__":
    main()
