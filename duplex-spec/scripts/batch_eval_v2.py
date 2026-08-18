"""Batch latency/overlap evaluation across multiple clips, in one run.

For each --start, generates baseline + the chosen trigger(s) audio, computes label-free
turn-taking metrics (energy VAD), and at the end aggregates mean +/- std across clips so the
result is a distribution, not a single cherry-picked clip. Also writes a per-clip CSV and a
latency-vs-overlap scatter (the 'controllable timing' figure).

Uses audio_demo_triggers.generate machinery indirectly by calling it as a subprocess per clip
(keeps this script simple and reuses the verified generation path).

Usage:
    PYTHONPATH=src python scripts/batch_eval.py \
        --tokens tokens_eval/<conv>.npy --feats pairs_eval/<conv>.npz \
        --labels handoff_labels_eval/<conv>.npz --head head_v0.pt --probe probe_top5.npz \
        --triggers baseline,probe --probe-thr 0.8 --arm-silence 3 \
        --starts 11200,13300,13350,1100,10800,8800,10850,4650 \
        --clip-frames 500 --device cuda --out batch_out/
"""
from __future__ import annotations
import argparse, subprocess, sys, json, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_metrics_v2 as am


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--triggers", default="baseline,probe")
    ap.add_argument("--probe-thr", default="0.8")
    ap.add_argument("--js-tau", default="0.2")
    ap.add_argument("--entropy-thr", default="0.6")
    ap.add_argument("--arm-silence", default="3")
    ap.add_argument("--temp", default="0.8")
    ap.add_argument("--starts", required=True, help="comma-separated start frames")
    ap.add_argument("--clip-frames", default="500")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vad-db", type=float, default=-40)
    ap.add_argument("--out", type=Path, default=Path("batch_out"))
    ap.add_argument("--demo-script", default="scripts/audio_demo_triggers.py")
    args = ap.parse_args()

    import soundfile as sf
    args.out.mkdir(parents=True, exist_ok=True)
    starts = [int(x) for x in args.starts.split(",")]
    trigs = args.triggers.split(",")
    cf = int(args.clip_frames)

    per_clip = []                       # list of dict(start, trigger, metrics...)
    for st in starts:
        cdir = args.out / f"start{st}"
        cmd = [sys.executable, args.demo_script,
               "--tokens", args.tokens, "--feats", args.feats, "--labels", args.labels,
               "--head", args.head, "--triggers", args.triggers,
               "--probe-thr", args.probe_thr, "--js-tau", args.js_tau,
               "--entropy-thr", args.entropy_thr, "--arm-silence", args.arm_silence,
               "--temp", args.temp, "--start", str(st), "--clip-frames", str(cf),
               "--device", args.device, "--out", str(cdir)]
        if args.probe:
            cmd += ["--probe", args.probe]
        env = {"PYTHONPATH": "src"}
        print(f"\n=== start {st} ===")
        r = subprocess.run(cmd, env={**__import__("os").environ, **env})
        if r.returncode != 0:
            print(f"[warn] generation failed at start {st}, skipping"); continue

        # find the generated wavs (clipNN_hM_<trigger>.wav) and score each
        wavs = sorted(cdir.glob("*.wav"))
        by_trig = {}
        for w in wavs:
            for t in trigs:
                if w.stem.endswith(f"_{t}"):
                    by_trig[t] = w
        for t in trigs:
            if t not in by_trig:
                continue
            wav, sr = sf.read(by_trig[t], always_2d=True)
            if wav.shape[1] < 2:
                continue
            u, fsec = am.vad(wav[:, 0], sr, thresh_db=args.vad_db)
            m, _ = am.vad(wav[:, 1], sr, thresh_db=args.vad_db)
            met = am.turn_stats(u, m, fsec)
            per_clip.append(dict(start=st, trigger=t, **met))
            print(f"  {t:>8}: engage={met['time_to_engage_ms']:.0f}ms "
                  f"gap_fill={met['gap_fill_pct']:.1f}% engage%={met['engage_rate_pct']:.1f} "
                  f"overlap={met['overlap_pct']:.1f}%")

    if not per_clip:
        sys.exit("no clips scored")

    # per-clip CSV
    csv_path = args.out / "per_clip.csv"
    keys = ["start", "trigger", "gap_fill_pct", "time_to_engage_ms", "engage_rate_pct",
            "overlap_pct", "dead_air_pct", "moshi_talk_pct", "n_moshi_turns", "n_turn_ends"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
        for row in per_clip:
            wr.writerow({k: row.get(k) for k in keys})

    # aggregate mean +/- std per trigger (also vs baseline delta)
    print("\n" + "="*66)
    print("AGGREGATE across clips (mean +/- std)")
    print(f"{'trigger':>10} {'gap_fill%':>16} {'overlap%':>14} {'engage%':>14} {'clips':>6}")
    print("-"*66)
    agg = {}
    base_gf = None
    for t in trigs:
        rows = [r for r in per_clip if r["trigger"] == t]
        if not rows: continue
        gf = np.array([r["gap_fill_pct"] for r in rows])
        ov = np.array([r["overlap_pct"] for r in rows])
        er = np.array([r["engage_rate_pct"] for r in rows])
        agg[t] = dict(gap_fill_mean=float(gf.mean()), gap_fill_std=float(gf.std()),
                      overlap_mean=float(ov.mean()), overlap_std=float(ov.std()),
                      engage_mean=float(er.mean()), engage_std=float(er.std()), n=len(rows))
        if t == "baseline": base_gf = agg[t]["gap_fill_mean"]
    for t in trigs:
        if t not in agg: continue
        a = agg[t]
        delta = ""
        if base_gf is not None and t != "baseline":
            d = a["gap_fill_mean"] - base_gf
            delta = f" ({'+' if d>=0 else ''}{d:.1f})"
        print(f"{t:>10} {a['gap_fill_mean']:>7.1f}±{a['gap_fill_std']:<5.1f}{delta:>8} "
              f"{a['overlap_mean']:>7.1f}±{a['overlap_std']:<4.1f} "
              f"{a['engage_mean']:>7.1f}±{a['engage_std']:<4.1f} {a['n']:>6}")

    (args.out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"\n[out] per-clip: {csv_path}")
    print(f"[out] aggregate: {args.out/'aggregate.json'}")

    # latency-vs-overlap scatter (the 'controllable timing' figure)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"baseline":"#888","probe":"#6A1B9A","js":"#1565C0","entropy":"#EF6C00"}
        plt.figure(figsize=(6,5))
        for t in trigs:
            rows=[r for r in per_clip if r["trigger"]==t]
            xs=[r["gap_fill_pct"] for r in rows]
            ys=[r["overlap_pct"] for r in rows]
            plt.scatter(xs,ys,label=t,c=colors.get(t,"#333"),s=60,alpha=0.7,edgecolors="white")
        plt.xlabel("gap-fill (%) — higher = more responsive")
        plt.ylabel("overlap (%) — lower = less barge-in")
        plt.title("Responsiveness vs overlap per clip")
        plt.legend(); plt.grid(alpha=0.3)
        fig=args.out/"tradeoff_scatter.png"; plt.savefig(fig,dpi=150,bbox_inches="tight")
        print(f"[out] scatter: {fig}")
    except Exception as e:
        print(f"  (scatter skipped: {e})")


if __name__ == "__main__":
    main()
