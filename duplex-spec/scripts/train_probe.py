"""Supervised stability probe --- predict whether a speculated frame will be ACCEPTABLE,
from features available at the FIRST prediction, so no m-frame wait is needed.

WHY THIS MATTERS (the ceiling argument)
---------------------------------------
The amendable gate needs a frame predicted m times before it is eligible: cols = K-(m-1),
so m=4 can never commit beyond horizon k=1 (80 ms) --- three quarters of the 320 ms horizon
is spent running the convergence test. A probe estimates "will this hold?" from features at
the first prediction, so every horizon becomes eligible and the full 320 ms is recoverable.

TARGET
------
Binary: is the true cb0 token in the CONFIDENT top-k of the head's distribution
(top-k AND normalised entropy < floor), per (frame, horizon, channel). This is the same
top-k acceptance the gates are scored under, so the probe lands on the SAME frontier.

FEATURES (all available at prediction time; temporal ones use earlier predictions of the
SAME absolute frame, sentinel-filled when no earlier prediction exists)
  instantaneous: norm-entropy, top1 prob, top1-top2 margin, horizon k (normalised)
  temporal:      JS(now, 1-ago), entropy delta, top1-prob delta, has_history flag

SPLIT
-----
Fit on a tuning split carved from the TRAINING conversations; evaluate the frontier on the
held-out 10. Never fit on held-out (would leak the comparison to the gates).

MODEL: logistic regression (interpretable; report coefficients). Threshold swept like tau.

Usage:
    # 1) extract features+labels from TRAIN conversations, fit the probe
    PYTHONPATH=src python scripts/train_probe.py fit \
        --head head_v0.pt --pairs-dir pairs_train/ --device cuda \
        --topk 5 --topk-ent-floor 0.5 --out probe_top5.npz

    # 2) evaluate the probe frontier on HELD-OUT conversations
    PYTHONPATH=src python scripts/train_probe.py eval \
        --head head_v0.pt --pairs-dir pairs_eval/ --device cuda \
        --probe probe_top5.npz --topk 5 --topk-ent-floor 0.5 \
        --thresholds 0.3,0.4,0.5,0.6,0.7,0.8,0.9
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np

FEAT_NAMES = ["norm_entropy", "top1_prob", "top1_top2_margin", "horizon_norm",
              "js_prev", "entropy_delta", "top1_delta", "has_history"]


def load_head(head_path, device):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from duplex_spec.head import MultiStepTPPHead, MultiStepDepHead
    ck = torch.load(head_path, map_location=device)
    K = ck["horizon"]
    Head = MultiStepDepHead if ck.get("head_type") == "dep" else MultiStepTPPHead
    head = Head(hidden_dim=ck["hidden_dim"], n_channels=ck["n_channels"],
                n_codebooks=ck["n_codebooks"], codebook_size=2048, horizon=K)
    head.load_state_dict(ck["state_dict"]); head.to(device).eval()
    return head, K, ck["n_channels"]


def js_pair(p, q):
    eps = 1e-8
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * (np.log(p) - np.log(m)), -1) + \
           0.5 * np.sum(q * (np.log(q) - np.log(m)), -1)


def extract_conv(head, K, C, feats, frames, tokens, batch, device, topk, ent_floor):
    """Return features [M,C,F], labels [M,C], and index bookkeeping for one conversation.
    M = number of (position,horizon) rows kept; features/labels are per channel.
    Temporal features compare horizon k at position i with horizon k+1 at position i-1
    (same absolute frame, one tick earlier)."""
    import torch
    logV = float(np.log(2048.0))
    Tlen = tokens.shape[2]
    rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + K < Tlen]
    if not rows:
        return None
    fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])

    P0, TR = [], []
    with torch.no_grad():
        for s in range(0, len(ft_row), batch):
            br = ft_row[s:s + batch]; bf = fr_idx[s:s + batch]
            x = torch.from_numpy(feats[br].astype(np.float32)).to(device)
            p = torch.softmax(head(x), dim=-1)
            P0.append(p[:, :, :, 0, :].cpu().numpy().astype(np.float32))    # [b,K,C,V]
            tru = np.stack([tokens[:, :, f + 1:f + 1 + K] for f in bf]).transpose(0, 3, 1, 2)
            TR.append(tru[:, :, :, 0].astype(np.int32))                     # [b,K,C] cb0 truth
    p0 = np.concatenate(P0); tru0 = np.concatenate(TR)                      # [N,K,C,V],[N,K,C]
    N = p0.shape[0]

    # instantaneous features
    srt = np.sort(p0, axis=-1)                                             # ascending
    top1 = srt[..., -1]; top2 = srt[..., -2]                               # [N,K,C]
    ent = -np.sum(np.clip(p0, 1e-8, 1) * np.log(np.clip(p0, 1e-8, 1)), -1) / logV
    margin = top1 - top2
    horizon = (np.arange(K)[None, :, None] / max(K - 1, 1)) * np.ones((N, K, C))

    # temporal features: same absolute frame, one tick earlier = (i-1, k+1)
    js_prev = np.full((N, K, C), 0.0); ent_d = np.zeros((N, K, C))
    top1_d = np.zeros((N, K, C)); has_hist = np.zeros((N, K, C))
    if N >= 2:
        a = p0[1:, : K - 1, :, :]                                          # (i,   k)
        b = p0[: N - 1, 1:K, :, :]                                         # (i-1, k+1) same frame
        js_prev[1:, : K - 1, :] = js_pair(a, b)
        ent_d[1:, : K - 1, :] = ent[1:, : K - 1, :] - ent[: N - 1, 1:K, :]
        top1_d[1:, : K - 1, :] = top1[1:, : K - 1, :] - top1[: N - 1, 1:K, :]
        has_hist[1:, : K - 1, :] = 1.0

    F = np.stack([ent, top1, margin, horizon, js_prev, ent_d, top1_d, has_hist], axis=-1)  # [N,K,C,8]

    # label: true cb0 token in confident top-k
    rank = (p0 > np.take_along_axis(p0, tru0[..., None], -1)).sum(-1)      # [N,K,C]
    label = ((rank < topk) & (ent < ent_floor)).astype(np.float32)        # [N,K,C]

    return F.reshape(-1, 8), label.reshape(-1), (N, K, C), p0, tru0


def cmd_fit(args):
    import torch
    head, K, C = load_head(args.head, args.device)
    pairs = [(z, z.with_suffix(".npy")) for z in sorted(args.pairs_dir.glob("*.npz"))
             if z.with_suffix(".npy").exists()]
    if not pairs:
        sys.exit("no pairs")
    Xs, ys = [], []
    for z, y in pairs:
        d = np.load(z); tk = np.load(y)
        r = extract_conv(head, K, C, d["feats"], d["frames"], tk,
                         args.batch, args.device, args.topk, args.topk_ent_floor)
        if r:
            Xs.append(r[0]); ys.append(r[1])
    X = np.concatenate(Xs); Y = np.concatenate(ys)
    print(f"[fit] {len(X)} rows, positive rate {Y.mean():.1%}")

    # standardise, then logistic regression (sklearn if present, else numpy GD)
    mu = X.mean(0); sd = X.std(0) + 1e-8
    Xs_ = (X - mu) / sd
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(Xs_, Y)
        w = clf.coef_[0].astype(np.float64); b = float(clf.intercept_[0])
        print("[fit] sklearn LogisticRegression")
    except Exception:
        w, b = _logreg_numpy(Xs_, Y)
        print("[fit] numpy logistic regression fallback")

    print("[coef] (standardised) " +
          "  ".join(f"{n}={c:+.3f}" for n, c in zip(FEAT_NAMES, w)))
    np.savez(args.out, w=w, b=b, mu=mu, sd=sd, feat_names=np.array(FEAT_NAMES),
             topk=args.topk, ent_floor=args.topk_ent_floor)
    print(f"[save] {args.out}")


def _logreg_numpy(X, y, iters=3000, lr=0.1):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    pos = y.mean(); cw = np.where(y > 0.5, 1.0 / max(pos, 1e-6), 1.0 / max(1 - pos, 1e-6))
    for _ in range(iters):
        z = X @ w + b; p = 1 / (1 + np.exp(-z))
        g = (p - y) * cw / n
        w -= lr * (X.T @ g); b -= lr * g.sum()
    return w, b


def cmd_eval(args):
    import torch
    head, K, C = load_head(args.head, args.device)
    pr = np.load(args.probe, allow_pickle=True)
    w, b, mu, sd = pr["w"], float(pr["b"]), pr["mu"], pr["sd"]
    pairs = [(z, z.with_suffix(".npy")) for z in sorted(args.pairs_dir.glob("*.npz"))
             if z.with_suffix(".npy").exists()]
    if not pairs:
        sys.exit("no pairs")
    thrs = [float(t) for t in args.thresholds.split(",")]

    # accumulate per-threshold totals + a leading-run commit over probe scores
    tot = {t: dict(r=0, n=0, roll=0, pts=0) for t in thrs}
    auc_s, auc_y = [], []
    for z, y in pairs:
        d = np.load(z); tk = np.load(y)
        r = extract_conv(head, K, C, d["feats"], d["frames"], tk,
                         args.batch, args.device, args.topk, args.topk_ent_floor)
        if not r:
            continue
        Xf, lab, (N, Kk, Cc), p0, tru0 = r
        score = 1 / (1 + np.exp(-(((Xf - mu) / sd) @ w + b)))             # [N*K*C]
        score = score.reshape(N, Kk, Cc)
        lab = lab.reshape(N, Kk, Cc)
        # acceptance mask (same top-k rule) for scoring precision/rollback
        accept = lab.astype(bool)                                         # label IS acceptance
        auc_s.append(score.reshape(-1)); auc_y.append(lab.reshape(-1))
        for t in thrs:
            commit = (score >= t).all(axis=2)                            # both channels pass
            clen = np.cumprod(commit, axis=1).sum(axis=1)                # leading run [N]
            acc_all = accept.all(axis=2)
            lead = np.cumprod(acc_all, axis=1).sum(axis=1)
            rr = np.minimum(clen, lead)
            T = tot[t]
            T["r"] += int(rr.sum()); T["n"] += int(clen.sum())
            T["roll"] += int((clen > rr).sum()); T["pts"] += len(clen)

    # AUC (rank quality of the probe score vs the acceptance label)
    s = np.concatenate(auc_s); yv = np.concatenate(auc_y)
    auc = _auc(s, yv)
    print(f"[probe] label=top{int(pr['topk'])} floor={float(pr['ent_floor'])}  AUC={auc:.3f}\n")
    print(f"{'threshold':>10} {'commit_prec':>12} {'rollback':>9} {'saved_ms':>9} {'committed':>10}")
    print("-" * 56)
    out = {"auc": auc}
    for t in thrs:
        T = tot[t]
        prec = T["r"] / T["n"] if T["n"] else float("nan")
        ro = T["roll"] / max(T["pts"], 1); sv = 80.0 * T["r"] / max(T["pts"], 1)
        co = T["n"] / max(T["pts"], 1)
        print(f"{t:>10.2f} {prec:>11.1%} {ro:>8.1%} {sv:>9.1f} {co:>10.3f}")
        out[f"thr{t}"] = dict(precision=prec, rollback=ro, saved_ms=sv, committed=co)
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"\n[json] {args.save_json}")
    print("Compare against the JS/argmax frontier: probe should reach horizons the")
    print("amendable gate cannot (k>1 at high m), i.e. more saved_ms at matched rollback.")


def _auc(scores, labels):
    order = np.argsort(scores); labels = labels[order]
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(labels) + 1)
    return (ranks[labels > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--head", type=Path, required=True)
        s.add_argument("--pairs-dir", type=Path, required=True)
        s.add_argument("--device", default="cuda")
        s.add_argument("--batch", type=int, default=512)
        s.add_argument("--topk", type=int, default=5)
        s.add_argument("--topk-ent-floor", type=float, default=0.5)
        if name == "fit":
            s.add_argument("--out", type=Path, default=Path("probe_top5.npz"))
        else:
            s.add_argument("--probe", type=Path, required=True)
            s.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
            s.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()
    (cmd_fit if args.cmd == "fit" else cmd_eval)(args)


if __name__ == "__main__":
    main()
