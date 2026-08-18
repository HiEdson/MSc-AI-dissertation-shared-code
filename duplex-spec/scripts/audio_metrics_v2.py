"""Turn-taking metrics focused on GAP-FILLING around user turn-ends (label-free, energy VAD).

Motivation: the old metric measured "user turn-end -> next Moshi silent->active transition",
which PENALISES a talkative/engaged system (if Moshi is already active at the turn-end, the
metric waits for it to stop and restart). This version instead credits engagement:

  time_to_engage_ms : ms until Moshi is active in the response window after a turn-end
                      (0 if already active); only counted where Moshi engages in-window.
  gap_fill_pct      : fraction of the post-turn-end window Moshi fills (HIGHER = more
                      responsive; this is the headline the ear actually perceives).
  engage_rate_pct   : fraction of user turn-ends Moshi responds to within the window.

Usage:
    python scripts/audio_metrics_v2.py --dir demo/ --clip clip00_h11700 --save-json m.json
    python scripts/audio_metrics_v2.py --audio a_base.wav a_probe.wav --labels baseline,probe
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def vad(x, sr, frame_ms=20, thresh_db=-40):
    fl = int(sr * frame_ms / 1000)
    if fl < 1 or len(x) < fl:
        return np.zeros(0, bool), frame_ms / 1000
    nf = len(x) // fl
    frames = x[:nf * fl].reshape(nf, fl)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    return db > thresh_db, fl / sr


def turn_stats(user_act, moshi_act, fsec, resp_window_ms=2000):
    n = min(len(user_act), len(moshi_act))
    u, m = user_act[:n], moshi_act[:n]
    both = u & m
    neither = (~u) & (~m)
    win = max(1, int(round(resp_window_ms / (fsec * 1000))))

    engage_times = []
    gap_fills = []
    n_ends = 0
    n_engaged = 0
    for t in range(1, n):
        if u[t - 1] and not u[t]:                       # user turn-end
            n_ends += 1
            w = m[t: min(n, t + win)]                   # Moshi activity in response window
            gap_fills.append(float(w.mean()) if len(w) else 0.0)
            active_idx = np.where(w)[0]
            if len(active_idx):
                n_engaged += 1
                engage_times.append(active_idx[0] * fsec * 1000)

    n_turns = int(np.sum(np.diff(m.astype(int)) == 1) + (1 if n and m[0] else 0))
    return dict(
        time_to_engage_ms=float(np.median(engage_times)) if engage_times else float("nan"),
        gap_fill_pct=100.0 * float(np.mean(gap_fills)) if gap_fills else 0.0,
        engage_rate_pct=100.0 * n_engaged / n_ends if n_ends else 0.0,
        overlap_pct=100.0 * both.sum() / n,
        dead_air_pct=100.0 * neither.sum() / n,
        moshi_talk_pct=100.0 * m.sum() / n,
        n_moshi_turns=n_turns,
        n_turn_ends=n_ends,
        response_latency_ms=float(np.median(engage_times)) if engage_times else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--clip")
    ap.add_argument("--audio", nargs="+")
    ap.add_argument("--labels")
    ap.add_argument("--thresh-db", type=float, default=-40)
    ap.add_argument("--resp-window-ms", type=float, default=2000)
    ap.add_argument("--save-json", type=Path)
    args = ap.parse_args()

    import soundfile as sf
    items = []
    if args.dir and args.clip:
        for f in sorted(args.dir.glob(f"{args.clip}_*.wav")):
            items.append((f.stem.replace(f"{args.clip}_", ""), f))
    elif args.audio:
        labs = (args.labels.split(",") if args.labels else [Path(a).stem for a in args.audio])
        items = list(zip(labs, [Path(a) for a in args.audio]))
    else:
        ap.error("give --dir with --clip, or --audio")

    rows = {}
    for label, path in items:
        wav, sr = sf.read(path, always_2d=True)
        if wav.shape[1] < 2:
            print(f"[skip] {path.name}: not stereo"); continue
        u, fsec = vad(wav[:, 0], sr, thresh_db=args.thresh_db)
        m, _ = vad(wav[:, 1], sr, thresh_db=args.thresh_db)
        rows[label] = turn_stats(u, m, fsec, resp_window_ms=args.resp_window_ms)

    if not rows:
        print("no stereo audio found"); return
    order = (["baseline"] if "baseline" in rows else []) + [k for k in rows if k != "baseline"]
    base = rows.get("baseline")

    print(f"\n{'condition':>10} {'engage_ms':>10} {'gap_fill%':>11} {'engage%':>9} "
          f"{'overlap%':>9} {'dead_air%':>10} {'turns':>6}")
    print("-" * 72)
    for k in order:
        m = rows[k]
        e = m["time_to_engage_ms"]; e_s = f"{e:.0f}" if e == e else "n/a"
        gd = ""
        if base and k != "baseline":
            d = m["gap_fill_pct"] - base["gap_fill_pct"]
            gd = f" ({'+' if d >= 0 else ''}{d:.1f})"
        print(f"{k:>10} {e_s:>10} {m['gap_fill_pct']:>9.1f}{gd:>8} {m['engage_rate_pct']:>8.1f} "
              f"{m['overlap_pct']:>8.1f} {m['dead_air_pct']:>9.1f} {m['n_moshi_turns']:>6}")
    print("\n  engage_ms  = median ms until Moshi engages after a user turn-end (0 if already")
    print("               active); only over turn-ends where it engages within the window.")
    print("  gap_fill%  = fraction of the post-turn-end window Moshi fills (HIGHER = more")
    print("               responsive; credits engagement instead of penalising it).")
    print("  engage%    = fraction of user turn-ends Moshi responds to within the window.")
    print("  Honest test: does the trigger RAISE gap_fill / engage% without RAISING overlap?")
    if args.save_json:
        args.save_json.write_text(json.dumps(rows, indent=2))
        print(f"\n[json] {args.save_json}")


if __name__ == "__main__":
    main()
