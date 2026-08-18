"""Moshi feasibility probe + cache-features extractor.

RUN THIS ON YOUR 16GB GPU. It answers two questions in one pass:
  1. Does Moshi load and stream within your VRAM?  (the hardware question)
  2. It caches per-frame hidden states (transformer_out) for head training.

NOTE: this was written against Moshi's documented API (the run_inference.py
patterns) but could NOT be executed in the assistant's sandbox (no GPU / no HF
download). Treat your first successful run as the real validation, and if an
API call differs, cross-check against `moshi/run_inference.py`.

Setup (in a moshi venv, NOT your duplex-spec venv):
    pip install moshi sphn
    python moshi_probe.py --hf-repo kyutai/moshiko-pytorch-q8 --seconds 10 --out feats.npy
    python moshi_probe.py --audio mychat.wav --out feats.npy        # real audio (24kHz)

Tip: q8 (~8GB) is the 16GB-friendly choice. If it OOMs, try q4. bf16 (~16GB)
will likely leave no headroom on a 16GB card.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def gb(x: float) -> float:
    return x / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", help="used for non-quantised repos")
    ap.add_argument("--audio", default=None, help="wav at 24kHz; else synthesises noise")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="save cached features here (.npy)")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("Need torch (CUDA build). Install for your CUDA version.")
    try:
        from moshi.models import LMGen, loaders
    except ImportError:
        sys.exit("Need moshi:  pip install moshi")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available — is this the CUDA torch build, on the GPU box?")

    def reset_peak():
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak():
        return torch.cuda.max_memory_allocated() if device == "cuda" else 0

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[gpu ] {props.name}  total={gb(props.total_memory):.1f} GB")

    # ---- Stage 1: load (the first place it can OOM) ----
    print(f"[load] {args.hf_repo} ...")
    reset_peak()
    try:
        ckpt = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
        mimi = ckpt.get_mimi(device=device)
        lm = ckpt.get_moshi(device=device, dtype=getattr(torch, args.dtype))
    except torch.cuda.OutOfMemoryError:
        sys.exit("OOM during LOAD — the model does not fit. Try q4, or request more VRAM.")
    except Exception as e:  # noqa: BLE001  (loader/quantisation surprises -> show clearly)
        sys.exit(f"Load failed ({type(e).__name__}: {e}). If using q8 and it errors on dtype, "
                 f"retry with --hf-repo kyutai/moshiko-pytorch-bf16 (needs ~16GB).")
    if device == "cuda":
        torch.cuda.synchronize()
    n_params = sum(p.numel() for p in lm.parameters())
    print(f"[load] params={n_params/1e9:.2f}B  peak={gb(peak()):.2f} GB")
    print(f"[cfg ] frame_rate={mimi.frame_rate}Hz sr={mimi.sample_rate} "
          f"codebooks={mimi.num_codebooks} channels={mimi.channels} dep_q={lm.dep_q}")

    # q8 fix: the loader casts ALL float tensors to --dtype, but QLinear's
    # `weight_scb` scale buffer must stay float32 (its forward rejects bf16).
    # Re-float the scale buffers on quantised models; harmless on bf16 models.
    try:
        from moshi.utils.quantize import QLinear
        n_fixed = 0
        for m in lm.modules():
            if isinstance(m, QLinear):
                m.weight_scb.data = m.weight_scb.data.float()
                n_fixed += 1
        if n_fixed:
            print(f"[q8  ] re-floated {n_fixed} QLinear scale buffers "
                  f"(note: bf16 round-trip is slightly lossy; load --dtype float32 "
                  f"if you need exact scales later)")
    except Exception:
        pass

    lm_gen = LMGen(lm)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)  # 1920 @ 24kHz/12.5Hz
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # ---- Stage 2: audio in ----
    if args.audio:
        import sphn
        wav, sr = sphn.read(args.audio)
        if sr != mimi.sample_rate:
            sys.exit(f"audio is {sr}Hz; resample to {mimi.sample_rate}Hz first.")
        wav = torch.from_numpy(wav).to(device).float()
        if wav.dim() == 1:
            wav = wav[None]
        in_pcm = wav[None, : mimi.channels]            # [1, channels, n]
    else:
        n = int(args.seconds * mimi.sample_rate)
        in_pcm = 0.01 * torch.randn(1, mimi.channels, n, device=device)

    chunks = [c for c in in_pcm.split(frame_size, dim=2) if c.shape[-1] == frame_size]
    if args.max_frames:
        chunks = chunks[: args.max_frames]
    print(f"[data] {len(chunks)} frames ({len(chunks) / mimi.frame_rate:.1f}s)")

    # ---- Stage 3: stream and capture transformer_out (the seam) ----
    # lm_gen._step returns (tokens, transformer_out) or None during the warmup delay.
    # We call _step directly because the public step() discards transformer_out.
    # (First-frame nuance from run_inference.py is ignored here; for training-grade
    #  caching, mirror run_inference or use lm.forward_text over a full sequence.)
    reset_peak()
    feats: list[np.ndarray] = []
    t0 = time.time()
    try:
        with torch.no_grad():
            for chunk in chunks:
                codes = mimi.encode(chunk)             # [1, num_codebooks, 1]
                res = lm_gen._step(codes)
                if res is None:
                    continue
                _tokens, transformer_out = res         # transformer_out: [1, 1, dim]
                feats.append(transformer_out[:, 0].float().cpu().numpy())  # [1, dim]
    except torch.cuda.OutOfMemoryError:
        sys.exit(f"OOM during STREAMING after {len(feats)} frames — inference exceeds VRAM. "
                 f"Try q4 or request more VRAM.")
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    if not feats:
        sys.exit("No features captured (all warmup). Increase --seconds.")
    feat = np.concatenate(feats, axis=0)               # [T, dim]
    audio_s = len(chunks) / mimi.frame_rate
    print(f"[run ] frames_kept={feat.shape[0]} dim={feat.shape[1]} "
          f"peak={gb(peak()):.2f} GB  wall={dt:.1f}s  realtime_x={audio_s/dt if dt else 0:.2f}")

    # ---- Stage 4: cache ----
    if args.out:
        np.save(args.out, feat)
        print(f"[save] {args.out}  ({feat.nbytes/1e6:.1f} MB, dtype={feat.dtype})")

    # ---- Verdict (your hardware answer) ----
    print("\n=== VERDICT ===")
    print(f"Moshi loads and streams on {device}: YES")
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"peak VRAM: {gb(peak()):.2f} GB / {gb(total):.1f} GB  "
              f"(headroom {gb(total - peak()):.2f} GB)")
        print("Head-training runs with the backbone UNLOADED on these cached features,")
        print("so if streaming fits, the whole project fits 16GB. If it OOM'd above,")
        print("that message is your evidence for a hardware request.")


if __name__ == "__main__":
    main()
