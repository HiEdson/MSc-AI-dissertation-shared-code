"""Plot results for the speculative-dialogue project.

Two kinds of figure, driven by JSON the eval / training scripts emit so plots
regenerate automatically as results improve:

  frontier  : commit-gate comparison (entropy vs amendable) — precision vs
              rollback, and latency-saved vs rollback. THE headline figure.
  curves    : per-horizon and per-codebook accuracy (head vs copy baseline).

Usage:
    python plot_results.py --eval-json eval_results.json --out figs/
    python plot_results.py --train-json train_results.json --out figs/
    python plot_results.py --eval-json e.json --train-json t.json --out figs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GATE = {
    "entropy":   dict(color="#c0392b", marker="o", label="Entropy (confidence)"),
    "amendable": dict(color="#2471a3", marker="s", label="Amendable (strict stability)"),
    "relaxed":   dict(color="#1e8449", marker="D", label="Amendable (relaxed k-of-m)"),
}


def _sorted_xy(points, xkey, ykey):
    pts = sorted(points, key=lambda p: p[xkey])
    return ([p[xkey] for p in pts], [p[ykey] for p in pts], [p.get("label", "") for p in pts])


def plot_frontier(results, accept, outdir, meta=None):
    data = results[accept]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for gate, pts in data.items():
        st = GATE.get(gate, dict(color="#555", marker="^", label=gate))
        kw = dict(color=st["color"], marker=st["marker"], markersize=7, linewidth=1.6)
        x, y, lab = _sorted_xy(pts, "rollback", "commit_prec")
        ax[0].plot([v * 100 for v in x], [v * 100 for v in y], label=st["label"], **kw)
        for xi, yi, li in zip(x, y, lab):
            ax[0].annotate(li, (xi * 100, yi * 100), fontsize=7,
                           xytext=(4, 4), textcoords="offset points")
        x2, y2, _ = _sorted_xy(pts, "rollback", "saved_ms")
        ax[1].plot([v * 100 for v in x2], y2, label=st["label"], **kw)
    ax[0].set_xlabel("Rollback rate (%)"); ax[0].set_ylabel("Commit precision (%)")
    ax[0].set_title("Precision vs rollback  (better \u2197 up-left)")
    ax[1].set_xlabel("Rollback rate (%)"); ax[1].set_ylabel("Latency hidden (ms / speculation)")
    ax[1].set_title("Latency saved vs rollback")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    sub = f"  (head val_loss={meta['val_loss']:.3f}, {meta.get('n_points','?')} pts, " \
          f"{meta.get('n_convs','?')} convs)" if meta else ""
    fig.suptitle(f"Commit-gate comparison — acceptance = {accept}{sub}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = outdir / f"frontier_{accept}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight"); paths.append(p)
    plt.close(fig)
    return paths


def plot_curves(train, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    # per-horizon: head vs copy
    if "per_horizon" in train:
        ph = train["per_horizon"]
        ks = list(range(1, len(ph["head"]) + 1))
        fig, a = plt.subplots(figsize=(5.2, 4))
        a.plot(ks, [v * 100 for v in ph["head"]], "o-", color="#2471a3", label="Head", linewidth=1.8)
        a.plot(ks, [v * 100 for v in ph["copy"]], "s--", color="#c0392b", label="Copy (persistence)", linewidth=1.8)
        a.set_xlabel("Prediction horizon (frames, 80 ms each)")
        a.set_ylabel("Token accuracy (%)"); a.set_xticks(ks)
        a.set_title("Prediction accuracy vs horizon"); a.grid(alpha=0.3); a.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = outdir / f"per_horizon.{ext}"; fig.savefig(p, dpi=150); paths.append(p)
        plt.close(fig)
    # per-codebook: head - copy margin
    if "per_codebook" in train:
        pc = train["per_codebook"]
        qs = list(range(len(pc["head"])))
        margin = [h - c for h, c in zip(pc["head"], pc["copy"])]
        fig, a = plt.subplots(figsize=(6, 4))
        colors = ["#2471a3" if m >= 0 else "#c0392b" for m in margin]
        a.bar([f"cb{q}" for q in qs], [m * 100 for m in margin], color=colors)
        a.axhline(0, color="k", linewidth=0.8)
        a.set_ylabel("Head \u2212 Copy accuracy (pp)")
        a.set_title("Where the head beats persistence (by codebook)")
        a.annotate("coarse / semantic\n(turn-taking)", (0, margin[0] * 100),
                   fontsize=8, xytext=(10, 10), textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", color="gray"))
        a.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = outdir / f"per_codebook.{ext}"; fig.savefig(p, dpi=150); paths.append(p)
        plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path)
    ap.add_argument("--train-json", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figs"))
    args = ap.parse_args()
    made = []
    if args.eval_json:
        blob = json.loads(args.eval_json.read_text())
        for accept in blob["results"]:
            made += plot_frontier(blob["results"], accept, args.out, blob.get("meta"))
    if args.train_json:
        made += plot_curves(json.loads(args.train_json.read_text()), args.out)
    if not made:
        raise SystemExit("Provide --eval-json and/or --train-json.")
    print("wrote:")
    for p in made:
        print(" ", p)


if __name__ == "__main__":
    main()
