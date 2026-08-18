"""A/B comparison mixer: combine two already-generated stereo clips (baseline, probe) into ONE
file where BOTH Moshi responses are audible --- one loud (foreground), one quiet (background) ---
over the shared user audio. Makes perceptual comparison easier than listening back-to-back.

Uses existing wavs only; nothing is regenerated. Writes two files: one with baseline loud + probe
quiet, one with probe loud + baseline quiet. Optionally pans the two conditions to opposite ears.

Usage:
    python scripts/mix_ab.py \
        --baseline demo/clip00_h11700_baseline.wav \
        --probe    demo/clip00_h11700_probe.wav \
        --quiet-gain 0.28 --out-dir demo_ab/ [--pan]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def load(path):
    import soundfile as sf
    w, sr = sf.read(path, always_2d=True)
    if w.shape[1] == 1:
        w = np.repeat(w, 2, axis=1)
    return w.astype(np.float32), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--quiet-gain", type=float, default=0.28,
                    help="loudness of the background condition (0-1); 0.28 ~ clearly quieter")
    ap.add_argument("--user-gain", type=float, default=0.9,
                    help="loudness of the shared user channel")
    ap.add_argument("--pan", action="store_true",
                    help="pan the two conditions to opposite ears (loud centre-ish, quiet other ear)")
    ap.add_argument("--out-dir", type=Path, default=Path("demo_ab"))
    args = ap.parse_args()

    import soundfile as sf
    b, sr = load(args.baseline)
    p, sr2 = load(args.probe)
    assert sr == sr2, "sample-rate mismatch"
    n = min(len(b), len(p))
    b, p = b[:n], p[:n]

    user = b[:, 0]                       # shared user (identical in both); take baseline's left
    moshi_b = b[:, 1]                    # baseline Moshi (right)
    moshi_p = p[:, 1]                    # probe Moshi (right)

    def norm(x):
        m = np.abs(x).max()
        return x / m if m > 0 else x
    user = norm(user) * args.user_gain
    moshi_b = norm(moshi_b)
    moshi_p = norm(moshi_p)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write_mix(loud, quiet, loud_name, quiet_name, fn):
        """loud/quiet are Moshi mono tracks; build a stereo file: user + loud + quiet*gain."""
        q = quiet * args.quiet_gain
        if args.pan:
            # user centred; loud slightly left, quiet fully right (so ears separate them)
            L = user + loud * 0.9 + q * 0.2
            R = user + loud * 0.5 + q * 1.0
            stereo = np.stack([L, R], 1)
        else:
            mono = user + loud + q
            stereo = np.stack([mono, mono], 1)      # same both ears
        stereo = stereo / max(np.abs(stereo).max(), 1e-6)
        sf.write(fn, stereo, sr)
        print(f"[out] {fn}  ({loud_name} loud, {quiet_name} quiet @ gain {args.quiet_gain})")

    write_mix(moshi_b, moshi_p, "baseline", "probe",
              args.out_dir / "AB_baseline_loud.wav")
    write_mix(moshi_p, moshi_b, "probe", "baseline",
              args.out_dir / "AB_probe_loud.wav")
    print(f"\nDuration {n/sr:.1f}s. Play the two files; the loud track is the focus, the quiet")
    print("one is the same moment from the other condition, for perceptual A/B comparison.")


if __name__ == "__main__":
    main()
