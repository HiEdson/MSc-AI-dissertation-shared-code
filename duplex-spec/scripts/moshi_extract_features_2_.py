"""Stage B: cache frozen-Moshi hidden states for the speculative head.

Feeds a Stage-A dual-channel token file [2, n_codebooks, T] through frozen Moshi
in TEACHER-FORCING mode and caches a per-frame hidden state. Those hidden states
are the INPUT features for the multi-step head; the dual-channel tokens are the
TARGETS.

How teacher-forcing works (from moshi/models/lm.py LMGen._step):
  - input_tokens          = speaker on the "user/listen" stream  -> [1, 8, 1]
  - depformer_replace_tokens = speaker on Moshi's "main" stream,
    which REPLACES Moshi's generated audio with real tokens       -> [1, 8, 1]

FEATURE-LAYER EXPERIMENT (--layer / --concat-layers)
  By default we cache `transformer_out` (the FINAL temporal hidden state). That
  representation is specialised for Moshi's own next-token generation, which may
  not be the best representation for ANTICIPATING the other speaker. This script
  can instead cache an EARLIER layer's output (or a concatenation of layers) via
  forward hooks, to test whether a different frozen feature lifts the frontier --
  all still frozen, so feature-caching stays fast.
    --layer final        cache transformer_out (DEFAULT; unchanged behaviour)
    --layer -2           cache output of the 2nd-to-last transformer layer (dim 4096)
    --layer 12           cache output of layer 12
    --concat-layers -1,-3,-6   concatenate several layers (dim = n*4096)
  IMPORTANT: train AND eval features for one experiment must use the SAME --layer
  spec (the head's input width and meaning depend on it). The chosen spec is
  stored in the .npz as `layer_spec`.

  First, discover the layer stack on YOUR Moshi version (paths differ by release):
    python moshi_extract_features.py --tokens any.npy --out /tmp/x.npz --list-layers
  If the default attribute path is wrong, pass the right one with --layers-attr.

Run on the GPU box (~0.5x realtime on a 16GB card):
    python moshi_extract_features.py --tokens tokens/<c>.npy --out feats/<c>.npz
    python moshi_extract_features.py --tokens tokens/<c>.npy --out feats_l-2/<c>.npz --layer -2

Output (.npz): feats [M, dim] float32, frames [M] int64, conv_id, layer_spec.

NOTE: the --layer hook path is written from the lm.py structure but NOT executed
in the assistant's sandbox. Run --list-layers first to confirm the module path,
and validate the first real run (no errors, M ~= T - max_delay, peak < 16GB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def gb(x: float) -> float:
    return x / 1e9


def resolve(obj, dotted: str):
    """Follow a dotted attribute path, e.g. 'transformer.layers'."""
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def take(h):
    """Layer/transformer output -> [B, dim] (handles [B,T,dim] streaming or [B,dim])."""
    return h[:, 0] if h.dim() == 3 else h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True, help="Stage-A .npy [2, K, T]")
    ap.add_argument("--out", type=Path, required=True, help="output .npz")
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--swap", action="store_true", help="swap which channel is the Moshi stream")
    ap.add_argument("--max-seconds", type=float, default=None, help="cap length for a quick test")
    # feature-layer experiment
    ap.add_argument("--layer", default="final",
                    help="'final' (transformer_out, default) or a layer index, e.g. -2, 12")
    ap.add_argument("--concat-layers", default=None,
                    help="comma list of layer indices to concatenate, e.g. '-1,-3,-6' (overrides --layer)")
    ap.add_argument("--layers-attr", default="transformer.layers",
                    help="dotted path from the LM to the layer ModuleList (confirm with --list-layers)")
    ap.add_argument("--list-layers", action="store_true",
                    help="load the model, print the transformer layer stack, and exit")
    args = ap.parse_args()

    try:
        import torch
        from moshi.models import LMGen, loaders
    except ImportError:
        sys.exit("Need torch + moshi in this env.")

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available.")

    # ---- load (with the q8 scale-buffer fix from the probe) ----
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ckpt = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    lm = ckpt.get_moshi(device=args.device, dtype=getattr(torch, args.dtype))
    try:
        from moshi.utils.quantize import QLinear
        for m in lm.modules():
            if isinstance(m, QLinear):
                m.weight_scb.data = m.weight_scb.data.float()
    except Exception:
        pass

    # ---- resolve the layer stack (for --list-layers and the hook experiment) ----
    try:
        layers = resolve(lm, args.layers_attr)
        n_layers = len(layers)
    except Exception as e:
        layers, n_layers = None, 0
        if args.list_layers or args.layer != "final" or args.concat_layers:
            print(f"[warn] could not resolve --layers-attr '{args.layers_attr}': {e}")
            print("[hint] candidate sub-modules of the LM (look for the transformer layer list):")
            for name, mod in lm.named_modules():
                if name.count(".") <= 2 and ("layer" in name.lower() or "transformer" in name.lower()):
                    print(f"        {name}: {type(mod).__name__}")
            sys.exit("Set --layers-attr to the dotted path of the layer ModuleList and retry.")

    if args.list_layers:
        print(f"[layers] '{args.layers_attr}' -> {n_layers} layers of {type(layers[0]).__name__}")
        print("        index a layer with --layer (e.g. -2) or --concat-layers (e.g. -1,-3,-6)")
        sys.exit(0)

    # ---- decide feature source ----
    if args.concat_layers:
        idxs = [int(x) % n_layers for x in args.concat_layers.split(",")]
        use_final, layer_spec = False, f"concat[{args.concat_layers}]"
    elif args.layer != "final":
        idxs = [int(args.layer) % n_layers]
        use_final, layer_spec = False, f"layer[{args.layer}]"
    else:
        idxs, use_final, layer_spec = [], True, "final"

    captured: dict = {}
    if not use_final:
        def make_hook(key):
            def hook(_m, _i, o):
                captured[key] = o[0] if isinstance(o, tuple) else o
            return hook
        for i in idxs:
            layers[i].register_forward_hook(make_hook(i))
        print(f"[feat] caching {layer_spec}  (hooks on layers {idxs} of {n_layers})")
    else:
        print(f"[feat] caching transformer_out (final)")

    # ---- prepare tokens ----
    tokens = np.load(args.tokens)                      # [2, K, T]
    assert tokens.ndim == 3 and tokens.shape[0] == 2, f"expected [2,K,T], got {tokens.shape}"
    _, K, T = tokens.shape
    if args.max_seconds:
        T = min(T, int(args.max_seconds * 12.5))
        tokens = tokens[:, :, :T]
    user_ch, moshi_ch = (1, 0) if args.swap else (0, 1)
    print(f"[data] {args.tokens.name}: K={K} T={T}  user=ch{user_ch} moshi=ch{moshi_ch}")

    needed = lm.num_codebooks - lm.dep_q - 1
    print(f"[lm  ] num_codebooks={lm.num_codebooks} dep_q={lm.dep_q} user_stream_codebooks={needed}")
    if K < needed:
        sys.exit(f"Stage-A gave {K} codebooks but Moshi wants {needed} on the user stream.")

    lm_gen = LMGen(lm)
    lm_gen.streaming_forever(1)
    dev = args.device

    def frame_tensor(ch: int, t: int) -> "torch.Tensor":
        return torch.from_numpy(tokens[ch, :, t]).to(dev).long()[None, :, None]  # [1,K,1]

    # ---- teacher-forced streaming ----
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    feats: list[np.ndarray] = []
    frames: list[int] = []
    try:
        with torch.no_grad():
            for t in range(T):
                user = frame_tensor(user_ch, t)[:, :needed]            # [1, needed, 1]
                moshi = frame_tensor(moshi_ch, t)[:, : lm.dep_q]       # [1, dep_q, 1]
                out = lm_gen._step(user, depformer_replace_tokens=moshi)
                if out is None:
                    continue                                           # warmup (delay) frames
                _tok, transformer_out = out
                if use_final:
                    h = take(transformer_out)                          # [1, dim]
                else:
                    parts = [take(captured[i]) for i in idxs]          # each [1, dim]
                    h = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
                feats.append(h.float().cpu().numpy())                  # [1, dim]
                frames.append(t)
    except torch.cuda.OutOfMemoryError:
        sys.exit(f"OOM after {len(feats)} frames — unexpected on 16GB; report this.")
    if dev == "cuda":
        torch.cuda.synchronize()

    if not feats:
        sys.exit("No features captured — sequence shorter than the model delay?")
    feat = np.concatenate(feats, axis=0)               # [M, dim]
    frames_arr = np.asarray(frames, dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, feats=feat, frames=frames_arr, conv_id=args.tokens.stem, layer_spec=layer_spec)

    peak = torch.cuda.max_memory_allocated() if dev == "cuda" else 0
    print(f"[ok  ] feats {feat.shape} ({layer_spec}) frames[{frames_arr[0]}..{frames_arr[-1]}]  "
          f"M={feat.shape[0]} (T={T}, warmup={T - feat.shape[0]})  peak={gb(peak):.2f} GB")
    print(f"[save] {args.out}  ({feat.nbytes/1e6:.1f} MB)")
    print("\nValidate: M ~= T - max_delay, peak < 16GB, no errors. Use the SAME --layer for train+eval.")


if __name__ == "__main__":
    main()
