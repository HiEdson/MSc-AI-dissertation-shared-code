"""Label-free turn-taking metrics from stereo demo audio (user=L, Moshi=R).

Compares conditions (baseline / js / probe / entropy / ...) purely on how the two AUDIO
channels behave --- no CANDOR handoff labels, since those are the humans' turns, not Moshi's.
For each stereo wav we detect per-channel voice activity (energy over a short window) and
compute:

  response_latency_ms : after the USER goes quiet (turn-end), how long until MOSHI starts.
                        Lower = faster response. THE headline number.
  overlap_pct         : fraction of time BOTH channels active (barge-in / talking over).
  user_silence_pct    : fraction of time NEITHER active after a user turn-end (dead air).
  moshi_talk_pct      : fraction of time Moshi is active at all.
  n_moshi_turns       : number of distinct Moshi speaking runs.

Compare a trigger against baseline: does it respond FASTER (lower latency) without MORE
overlap? That is the honest test --- and note gating may NOT beat vanilla Moshi, since
releasing the gate doesn't pre-fill content.

Usage:
    python scripts/audio_metrics.py --dir demo_trigger_audio/ --clip clip01_h9684
    # or explicit files:
    python scripts/audio_metrics.py --audio a_baseline.wav a_js.wav --labels baseline,js
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np


def vad(x, sr, frame_ms=20, thresh_db=-40):
    """Simple energy VAD -> per-frame boolean active. Returns (active[F], frame_sec)."""
    fl = int(sr * frame_ms / 1000)
    if fl < 1 or len(x) < fl:
        return np.zeros(0, bool), frame_ms / 1000
    nf = len(x) // fl
    frames = x[:nf * fl].reshape(nf, fl)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    return db > thresh_db, fl / sr


def turn_stats(user_act, moshi_act, fsec):
    """Compute turn-taking metrics from two boolean activity tracks."""
    n = min(len(user_act), len(moshi_act))
    u, m = user_act[:n], moshi_act[:n]
    both = u & m
    neither = (~u) & (~m)

    # response latency: at each user turn-END (user active -> inactive), time until moshi active
    lat = []
    for t in range(1, n):
        if u[t - 1] and not u[t]:                 # user just went quiet
            j = t
            while j < n and not m[j]:
                j += 1
            if j < n:
                lat.append((j - t) * fsec * 1000)  # ms until moshi speaks
    # moshi turns
    n_turns = int(np.sum(np.diff(m.astype(int)) == 1) + (1 if n and m[0] else 0))
    return dict(
        response_latency_ms=float(np.median(lat)) if lat else float("nan"),
        response_latency_mean_ms=float(np.mean(lat)) if lat else float("nan"),
        overlap_pct=100.0 * both.sum() / n,
        dead_air_pct=100.0 * neither.sum() / n,
        moshi_talk_pct=100.0 * m.sum() / n,
        n_moshi_turns=n_turns,
        n_responses=len(lat),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, help="demo audio dir")
    ap.add_argument("--clip", help="clip prefix, e.g. clip01_h9684 (compares all its triggers)")
    ap.add_argument("--audio", nargs="+", help="explicit stereo wavs")
    ap.add_argument("--labels", help="comma labels matching --audio")
    ap.add_argument("--thresh-db", type=float, default=-40)
    ap.add_argument("--save-json", type=Path)
    args = ap.parse_args()

    import soundfile as sf
    items = []                                    # (label, path)
    if args.dir and args.clip:
        for f in sorted(args.dir.glob(f"{args.clip}_*.wav")):
            trig = f.stem.replace(f"{args.clip}_", "")
            items.append((trig, f))
    elif args.audio:
        labs = (args.labels.split(",") if args.labels
                else [Path(a).stem for a in args.audio])
        items = list(zip(labs, [Path(a) for a in args.audio]))
    else:
        ap.error("give --dir with --clip, or --audio")

    rows = {}
    for label, path in items:
        wav, sr = sf.read(path, always_2d=True)
        if wav.shape[1] < 2:
            print(f"[skip] {path.name}: not stereo"); continue
        u_act, fsec = vad(wav[:, 0], sr, thresh_db=args.thresh_db)
        m_act, _ = vad(wav[:, 1], sr, thresh_db=args.thresh_db)
        rows[label] = turn_stats(u_act, m_act, fsec)

    if not rows:
        print("no stereo audio found"); return
    # order: baseline first if present
    order = (["baseline"] if "baseline" in rows else []) + \
            [k for k in rows if k != "baseline"]

    print(f"\n{'condition':>10} {'resp_lat_ms':>12} {'overlap%':>9} {'dead_air%':>10} "
          f"{'moshi_talk%':>12} {'turns':>6}")
    print("-" * 64)
    base = rows.get("baseline")
    for k in order:
        m = rows[k]
        lat = m["response_latency_ms"]
        lat_s = f"{lat:.0f}" if lat == lat else "n/a"     # nan check
        delta = ""
        if base and k != "baseline" and lat == lat and base["response_latency_ms"] == base["response_latency_ms"]:
            d = lat - base["response_latency_ms"]
            delta = f" ({'+' if d>=0 else ''}{d:.0f} vs base)"
        print(f"{k:>10} {lat_s:>12}{delta} {m['overlap_pct']:>8.1f} "
              f"{m['dead_air_pct']:>9.1f} {m['moshi_talk_pct']:>11.1f} {m['n_moshi_turns']:>6}")
    print("\n  resp_lat_ms = median ms from user turn-end to Moshi starting (LOWER = faster).")
    print("  overlap%    = both talking (barge-in). dead_air% = neither (silence).")
    print("  Honest test vs baseline: does the trigger LOWER latency without RAISING overlap?")
    if args.save_json:
        args.save_json.write_text(json.dumps(rows, indent=2))
        print(f"\n[json] {args.save_json}")


if __name__ == "__main__":
    main()
