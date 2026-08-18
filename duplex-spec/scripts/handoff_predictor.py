"""Handoff predictor --- learn WHEN the conversational floor changes hands, from the
channel-activity stream alone. No head, no token speculation: this predicts the TIMING
event (a turn transition) that the whole project is ultimately about.

Idea
----
Moshi already knows WHAT to say. The open problem is WHEN. If we can predict "a handoff
is coming within H frames" early and reliably, that signal triggers response generation in
advance, hiding the reactive latency floor. This sidesteps the hard 2048-way token target
and the independent-head bottleneck entirely.

Target
------
From each channel's cb0 stream, active = (cb0 != that channel's silence token). A handoff
at frame t = the currently-active speaker stops AND the currently-silent one starts, within
the next H frames. Label_t = 1 if a handoff occurs in (t, t+H].

Features (windowed over the last N frames --- the sequential signal an LR can read)
  - both channels' current activity
  - current speaker's turn length so far (turns don't end instantly)
  - recent silence fraction per channel over last N (is the speaker trailing off?)
  - number of short pauses in the window (mid-turn vs turn-final rhythm)
  - activity trend: silence fraction in the last N/2 minus the first N/2 (accelerating stop)

Early-warning metric (the point)
--------------------------------
Not just accuracy. For each LEAD time l in {1..H} frames, precision/recall of "handoff in
exactly l frames": how early can we fire while keeping false alarms (barge-ins) low? The
latency hidden ~ how many ms ahead we can reliably fire.

Usage:
    # fit on training conversations
    PYTHONPATH=src python scripts/handoff_predictor.py fit \
        --pairs-dir pairs/ --N 12 --H 4 --out handoff_lr.npz
    # evaluate early-warning curve on held-out
    PYTHONPATH=src python scripts/handoff_predictor.py eval \
        --pairs-dir pairs_eval/ --model handoff_lr.npz --N 12 --H 4
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np

FEATS = ["act_ch0", "act_ch1", "turnlen_norm", "sil_frac_ch0", "sil_frac_ch1",
         "n_pauses", "trend_ch0", "trend_ch1", "who_active"]


def silence_ids(cb0):
    """Per-channel silence token = modal cb0 value. cb0: [C, T]."""
    sils = []
    for c in range(cb0.shape[0]):
        vals, counts = np.unique(cb0[c], return_counts=True)
        sils.append(int(vals[counts.argmax()]))
    return sils


def activity(tokens):
    """[C, T] boolean active mask from cb0 vs per-channel silence token."""
    cb0 = tokens[:, 0, :]                                  # [C, T]
    sils = silence_ids(cb0)
    return np.stack([cb0[c] != sils[c] for c in range(cb0.shape[0])]), sils


def build_labels(act, H):
    """act: [2, T] bool. Handoff at t = active speaker stops & other starts within (t, t+H].
    Returns label[T] (1 if any handoff in the horizon) and lead[T] (frames to next handoff,
    or -1). A handoff frame h is where the 'dominant' channel flips."""
    C, T = act.shape
    # dominant channel per frame: whichever is active; ties/both-silent -> carry previous
    dom = np.full(T, -1)
    cur = -1
    for t in range(T):
        a0, a1 = act[0, t], act[1, t]
        if a0 and not a1:
            cur = 0
        elif a1 and not a0:
            cur = 1
        # both active or both silent -> keep cur (overlap / gap)
        dom[t] = cur
    # handoff frames: dominant flips from one speaker to the other (ignore -1)
    ho = np.zeros(T, bool)
    for t in range(1, T):
        if dom[t] != -1 and dom[t - 1] != -1 and dom[t] != dom[t - 1]:
            ho[t] = True
    ho_idx = np.where(ho)[0]
    label = np.zeros(T, np.float32); lead = np.full(T, -1)
    for t in range(T):
        nxt = ho_idx[(ho_idx > t) & (ho_idx <= t + H)]
        if len(nxt):
            label[t] = 1.0; lead[t] = int(nxt[0] - t)
    return label, lead, dom


def build_features(act, dom, N):
    """Windowed features over the last N frames. Returns X[T, F] (rows with <N history
    are still emitted with a short window; valid mask flags full-window rows)."""
    C, T = act.shape
    X = np.zeros((T, len(FEATS)), np.float32)
    valid = np.zeros(T, bool)
    # turn length so far: frames since the last dominance change
    turnlen = np.zeros(T)
    tl = 0
    for t in range(T):
        if t > 0 and dom[t] != dom[t - 1] and dom[t] != -1:
            tl = 0
        tl += 1; turnlen[t] = tl
    for t in range(T):
        lo = max(0, t - N + 1); w = act[:, lo:t + 1]                # [2, <=N]
        n = w.shape[1]
        half = max(1, n // 2)
        first, second = w[:, :half], w[:, half:]
        if second.shape[1] == 0:
            second = first
        sil0 = 1.0 - w[0].mean(); sil1 = 1.0 - w[1].mean()
        # pauses: transitions active->silent in the dominant channel within window
        d = dom[t] if dom[t] != -1 else 0
        wc = w[d]
        n_pauses = int(np.sum((wc[:-1]) & (~wc[1:]))) if n > 1 else 0
        trend0 = first[0].mean() - second[0].mean()
        trend1 = first[1].mean() - second[1].mean()
        X[t] = [float(act[0, t]), float(act[1, t]), min(turnlen[t] / 50.0, 2.0),
                sil0, sil1, min(n_pauses / 5.0, 2.0), trend0, trend1, float(d)]
        valid[t] = (t >= N - 1) and (dom[t] != -1)
    return X, valid


def labels_from_transcript(labels_dir, conv_id, T, H):
    """Load real handoff frames for conv_id and expand to per-frame label/lead over horizon H.
    Returns (label[T], lead[T], taker[T]) or None if no label file."""
    fp = Path(labels_dir) / f"{conv_id}.npz"
    if not fp.exists():
        return None
    d = np.load(fp)
    ho = d["handoff_frames"].astype(int); tk = d["taker_channel"].astype(int)
    ho = ho[ho < T]
    label = np.zeros(T, np.float32); lead = np.full(T, -1); taker = np.full(T, -1)
    for h, who in zip(ho, tk):
        lo = max(0, h - H)
        for t in range(lo, h):                       # frames within H before the handoff
            if lead[t] == -1 or (h - t) < lead[t]:
                label[t] = 1.0; lead[t] = h - t; taker[t] = who
    return label, lead, taker


def conv_iter(pairs_dir):
    for npy in sorted(Path(pairs_dir).glob("*.npy")):
        yield npy, np.load(npy)


def cmd_fit(args):
    Xs, ys = [], []
    for npy, tokens in conv_iter(args.pairs_dir):
        act, _ = activity(tokens)
        _, _, dom = build_labels(act, args.H)                    # dom still needed for features
        X, valid = build_features(act, dom, args.N)
        if args.labels_dir:
            r = labels_from_transcript(args.labels_dir, npy.stem, tokens.shape[2], args.H)
            if r is None:
                continue
            label = r[0]
        else:
            label, _, _ = build_labels(act, args.H)
        Xs.append(X[valid]); ys.append(label[valid])
    X = np.concatenate(Xs); Y = np.concatenate(ys)
    print(f"[fit] {len(X)} frames, handoff-imminent rate {Y.mean():.1%}")
    mu = X.mean(0); sd = X.std(0) + 1e-8
    Xs_ = (X - mu) / sd
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xs_, Y); w = clf.coef_[0]; b = float(clf.intercept_[0])
        print("[fit] sklearn LogisticRegression (balanced)")
    except Exception:
        w, b = _lr_numpy(Xs_, Y); print("[fit] numpy fallback")
    print("[coef] " + "  ".join(f"{n}={c:+.3f}" for n, c in zip(FEATS, w)))
    np.savez(args.out, w=w, b=b, mu=mu, sd=sd, N=args.N, H=args.H,
             feats=np.array(FEATS))
    print(f"[save] {args.out}")


def _lr_numpy(X, y, iters=3000, lr=0.1):
    n, d = X.shape; w = np.zeros(d); b = 0.0
    pos = max(y.mean(), 1e-6)
    cw = np.where(y > 0.5, 1.0 / pos, 1.0 / (1 - pos))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b))); g = (p - y) * cw / n
        w -= lr * (X.T @ g); b -= lr * g.sum()
    return w, b


def cmd_eval(args):
    m = np.load(args.model, allow_pickle=True)
    w, b, mu, sd = m["w"], float(m["b"]), m["mu"], m["sd"]
    H = int(m["H"]); N = int(m["N"])
    # collect scores + lead-to-next-handoff for every valid frame
    S, LEAD, Y = [], [], []
    for npy, tokens in conv_iter(args.pairs_dir):
        act, _ = activity(tokens)
        _, lead_h, dom = build_labels(act, H)
        X, valid = build_features(act, dom, N)
        if args.labels_dir:
            r = labels_from_transcript(args.labels_dir, npy.stem, tokens.shape[2], H)
            if r is None:
                continue
            label, lead = r[0], r[1]
        else:
            label, lead, _ = build_labels(act, H)
        sc = 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b)))
        S.append(sc[valid]); LEAD.append(lead[valid]); Y.append(label[valid])
    S = np.concatenate(S); LEAD = np.concatenate(LEAD); Y = np.concatenate(Y)
    auc = _auc(S, Y)
    print(f"[handoff] N={N} H={H}  positive rate={Y.mean():.1%}  AUC={auc:.3f}\n")

    # early-warning: at a fixed operating threshold, precision & recall of firing, and
    # the distribution of how early we fire before the actual handoff.
    print(f"{'threshold':>9} {'precision':>10} {'recall':>8} {'fire_rate':>10} {'mean_lead_ms':>13}")
    print("-" * 54)
    out = {"auc": auc}
    for thr in [float(t) for t in args.thresholds.split(",")]:
        fire = S >= thr
        tp = int((fire & (Y > 0.5)).sum()); fp = int((fire & (Y < 0.5)).sum())
        fn = int((~fire & (Y > 0.5)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        # mean lead (ms) among correct early fires
        lead_hit = LEAD[(fire) & (Y > 0.5) & (LEAD > 0)]
        mean_lead = 80.0 * lead_hit.mean() if len(lead_hit) else 0.0
        print(f"{thr:>9.2f} {prec:>9.1%} {rec:>7.1%} {fire.mean():>9.1%} {mean_lead:>12.0f}")
        out[f"thr{thr}"] = dict(precision=prec, recall=rec, fire_rate=float(fire.mean()),
                                mean_lead_ms=mean_lead)
    print("\n  precision = of the times we fire, how often a handoff really comes (1-precision")
    print("             = barge-in / false-alarm rate).")
    print("  recall    = of real handoffs, how many we caught early enough.")
    print("  mean_lead = how many ms before the handoff we fired = latency hidden.")
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"[json] {args.save_json}")


def _auc(s, y):
    o = np.argsort(s); y = y[o]; npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return (ranks[y > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--pairs-dir", type=Path, required=True)
        s.add_argument("--N", type=int, default=12, help="feature window (frames)")
        s.add_argument("--H", type=int, default=4, help="handoff horizon (frames)")
        s.add_argument("--labels-dir", type=Path, default=None,
                       help="dir of candor_handoffs .npz (real turn boundaries). If set, "
                            "labels come from transcripts instead of the silence heuristic.")
        if name == "fit":
            s.add_argument("--out", type=Path, default=Path("handoff_lr.npz"))
        else:
            s.add_argument("--model", type=Path, required=True)
            s.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
            s.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()
    (cmd_fit if args.cmd == "fit" else cmd_eval)(args)


if __name__ == "__main__":
    main()
