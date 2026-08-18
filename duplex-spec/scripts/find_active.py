"""Find the most ACTIVE (real speech) stretches of a conversation, using ENERGY-based VAD on
the decoded audio --- not the token-modal test.

WHY energy, not tokens: in a full-duplex codec both channels ALWAYS carry signal (room tone,
breathing, background), so the tokens are always changing. "token != silence-token" therefore
flags ~90% of frames as 'active' even when nobody is speaking words. Energy (RMS of the decoded
waveform) distinguishes actual speech (loud) from ambient carry (quiet), which is what we want
for turn-taking.

Decodes each channel to audio via Mimi, computes per-frame RMS energy, thresholds it, and
reports the windows with the most USER speech.

Usage:
    PYTHONPATH=src python scripts/find_active.py \
        --tokens tokens_eval/<conv>.npy --win 200 --top 8 --device cuda
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--win", type=int, default=200)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--thresh-db", type=float, default=-40,
                    help="RMS threshold (dB) above which a frame counts as speech")
    args = ap.parse_args()

    import torch
    from moshi.models import loaders
    print("[load] Mimi ...")
    ckpt = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    mimi = ckpt.get_mimi(device=args.device)

    tok = np.load(args.tokens)
    C, Q, T = tok.shape
    sr = mimi.sample_rate
    fps = 12.5
    spf = int(sr / fps)                             # audio samples per 80ms frame

    def speech_activity(ch, chunk=1500):
        """Decode channel in CHUNKS (avoid OOM) -> per-FRAME speech flag via RMS energy. [T]."""
        flags = []
        with torch.no_grad():
            for s in range(0, T, chunk):
                e = min(T, s + chunk)
                seg = torch.from_numpy(tok[ch:ch+1, :, s:e]).to(args.device).long()
                wav = mimi.decode(seg).squeeze().cpu().numpy()
                nf = len(wav) // spf
                frames = wav[:nf*spf].reshape(nf, spf)
                rms = np.sqrt((frames**2).mean(1) + 1e-12)
                db = 20*np.log10(rms + 1e-12)
                flags.append((db > args.thresh_db).astype(np.float32))
                del seg
                if args.device == "cuda":
                    torch.cuda.empty_cache()
        act = np.concatenate(flags)
        return act, len(act)

    print("[decode] channel 0 (user) ...")
    u, nfu = speech_activity(0)
    print("[decode] channel 1 (other) ...")
    m, nfm = speech_activity(1) if C > 1 else (np.zeros(T), T)
    n = min(len(u), len(m))
    u, m = u[:n], m[:n]

    print(f"\nconversation: {n} frames ({80*n/1000:.0f}s)")
    print(f"USER speaking: {u.mean():.1%} of frames  |  other channel: {m.mean():.1%}")
    print("(energy-based; much lower than the token test because it ignores ambient carry)\n")

    rows = []
    for s in range(0, n - args.win, args.stride):
        e = s + args.win
        uu = u[s:e].mean(); mm = m[s:e].mean()
        both = ((u[s:e] > 0.5) & (m[s:e] > 0.5)).mean()
        # a good demo clip: user talks a lot AND there's turn-taking (some but not total overlap)
        rows.append((s, e, uu, mm, both))
    rows.sort(key=lambda r: -r[2])

    print(f"{'start':>7} {'end':>7} {'user_speech':>12} {'other':>7} {'overlap':>8}  suggested")
    print("-" * 66)
    for s, e, uu, mm, both in rows[:args.top]:
        print(f"{s:>7} {e:>7} {uu:>11.1%} {mm:>6.1%} {both:>7.1%}  "
              f"--start {s} --clip-frames {args.win}")
    print("\n  user_speech = fraction of the window where the USER is actually speaking (energy).")
    print("  Pick a window with high user_speech and MODERATE overlap (real turn-taking, not")
    print("  constant talk-over) for the clearest demo.")


if __name__ == "__main__":
    main()
