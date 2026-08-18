"""Stage B: cache frozen-Moshi hidden states for the speculative head.

Feeds a Stage-A dual-channel token file [2, n_codebooks, T] through frozen Moshi
in TEACHER-FORCING mode and caches `transformer_out` (the 4096-d temporal hidden
state) per frame. Those hidden states are the INPUT features for the multi-step
head; the dual-channel tokens themselves are the TARGETS.

How teacher-forcing works (from moshi/models/lm.py LMGen._step):
  - input_tokens          = speaker on the "user/listen" stream  -> [1, 8, 1]
  - depformer_replace_tokens = speaker on Moshi's "main" stream,
    which REPLACES Moshi's generated audio with real tokens       -> [1, 8, 1]
  So both real speakers drive the model; we read transformer_out each frame.

Two modelling choices (documented, revisit later):
  1. Channel assignment: by default L (channel 0) = user stream, R (channel 1) =
     Moshi stream. Use --swap to flip. The task is symmetric (predict both), so
     running both assignments is also a free data-augmentation later.
  2. The TEXT / inner-monologue stream is still SAMPLED by Moshi (there is no
     ground-truth text for CANDOR speakers). This is the one out-of-distribution
     element; a later option is to suppress/force it.

Run on the GPU box (same env as moshi_probe; ~0.5x realtime on a 16GB card):
    python moshi_extract_features.py --tokens tokens/<conv>.npy --out feats/<conv>.npz
    python moshi_extract_features.py --tokens tokens/<conv>.npy --out feats/<conv>.npz --max-seconds 60

Output (.npz): feats [M, dim] float32, frames [M] int64 (the frame index each
feature corresponds to, after warmup), conv_id.

NOTE: written from the lm.py source but NOT executed in the assistant's sandbox.
Validate on first run: no errors, M ~= T - max_delay, peak VRAM < 16GB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def gb(x: float) -> float:
    return x / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True, help="Stage-A .npy [2, K, T]")
    ap.add_argument("--out", type=Path, required=True, help="output .npz")
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--swap", action="store_true", help="swap which channel is the Moshi stream")
    ap.add_argument("--max-seconds", type=float, default=None, help="cap length for a quick test")
    args = ap.parse_args()

    try:
        import torch
        from moshi.models import LMGen, loaders
    except ImportError:
        sys.exit("Need torch + moshi in this env.")

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available.")

    tokens = np.load(args.tokens)                      # [2, K, T]
    assert tokens.ndim == 3 and tokens.shape[0] == 2, f"expected [2,K,T], got {tokens.shape}"
    _, K, T = tokens.shape
    if args.max_seconds:
        T = min(T, int(args.max_seconds * 12.5))
        tokens = tokens[:, :, :T]
    user_ch, moshi_ch = (1, 0) if args.swap else (0, 1)
    print(f"[data] {args.tokens.name}: K={K} T={T}  user=ch{user_ch} moshi=ch{moshi_ch}")

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

    needed = lm.num_codebooks - lm.dep_q - 1
    print(f"[lm  ] num_codebooks={lm.num_codebooks} dep_q={lm.dep_q} "
          f"user_stream_codebooks={needed}")
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
                feats.append(transformer_out[:, 0].float().cpu().numpy())  # [1, dim]
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
    np.savez(args.out, feats=feat, frames=frames_arr, conv_id=args.tokens.stem)

    peak = torch.cuda.max_memory_allocated() if dev == "cuda" else 0
    print(f"[ok  ] feats {feat.shape} frames[{frames_arr[0]}..{frames_arr[-1]}]  "
          f"M={feat.shape[0]} (T={T}, warmup={T - feat.shape[0]})  peak={gb(peak):.2f} GB")
    print(f"[save] {args.out}  ({feat.nbytes/1e6:.1f} MB)")
    print("\nValidate: M ~= T - max_delay, dim=4096, peak < 16GB, no errors.")


if __name__ == "__main__":
    main()
