"""Speculative evaluation: ENTROPY gate vs AMENDABLE (stability) gate.

Computes each frame's head prediction + per-step entropy once (batched), then
scores two commit criteria on identical data so they can be compared directly:

  entropy gate    - commit leading frames while the head's output entropy stays
                    below a threshold ("commit while confident"). Irrevocable /
                    Wald-style: one-shot confidence, no use of accumulating input.
  amendable gate  - commit a frame only when its prediction has CONVERGED across
                    the last m vantage points (horizon k now == horizon k+1 one
                    step ago == ...): "commit when the tentative decision stops
                    being amended as evidence arrives." Larger m = stricter.

Metrics (per acceptance criterion cb0 / exact):
  commit_prec  fraction of pre-fired frames correct
  rollback     fraction of speculations that mispredicted (would glitch)
  saved_ms     mean latency hidden per speculation (correct pre-fires only)
  committed    mean frames pre-fired per speculation

Usage:
    python eval_speculative.py --head head.pt --pairs-dir pairs_eval/ --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from duplex_spec.spec_eval import stability_commit_lengths, relaxed_commit_lengths  # noqa: E402

MS_PER_FRAME = 80.0


def leading_run(b: np.ndarray) -> np.ndarray:
    """[N,K] bool -> [N] count of leading True per row."""
    return np.cumprod(b, axis=1).sum(axis=1).astype(int)


def frame_acceptable_arr(pred, truth, accept, frac, topk_masks=None):
    """Per-(position, horizon) acceptability under an accept rule -> [N, K] bool.

    topk_masks: optional dict {name: bool[N,K]} of precomputed acceptability for
    distribution-based rules (top-k of the cb0 head distribution + entropy floor),
    computed in pass 1 where the softmax is available.
    """
    if accept == "cb0":
        return (pred[:, :, :, 0] == truth[:, :, :, 0]).all(axis=2)
    if accept == "exact":
        return (pred == truth).all(axis=(2, 3))
    if accept.startswith("top"):
        if topk_masks is None or accept not in topk_masks:
            raise ValueError(f"{accept} requires a precomputed mask")
        return topk_masks[accept]
    return (pred == truth).mean(axis=(2, 3)) >= frac                    # frac


def leading_acceptable(pred, truth, accept, frac, topk_masks=None):
    return leading_run(frame_acceptable_arr(pred, truth, accept, frac, topk_masks))


def metrics(commit_len, lead):
    acceptable = np.minimum(commit_len, lead)
    n = len(commit_len)
    tot_c = commit_len.sum()
    prec = acceptable.sum() / tot_c if tot_c else 1.0
    rollback = (commit_len > acceptable).mean()
    saved = acceptable.mean() * MS_PER_FRAME
    return prec, rollback, saved, commit_len.mean()


def row(label, m):
    prec, rb, saved, comm = m
    return f"{label:>10} {prec:>11.1%} {rb:>9.1%} {saved:>8.0f} {comm:>10.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--feats", type=Path, action="append", default=[])
    ap.add_argument("--tokens", type=Path, action="append", default=[])
    ap.add_argument("--pairs-dir", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--thresholds", default="0.3,0.5,0.7,0.85,0.95")
    ap.add_argument("--stability-m", default="2,3,4")
    ap.add_argument("--relaxed", default="3:2,4:3,4:2", help="window:min_agree pairs for the k-of-m relaxed gate")
    ap.add_argument("--relaxed-ent-floor", type=float, default=None, help="optional entropy floor added to every relaxed variant")
    ap.add_argument("--accept", default="cb0,exact,top5,top10")
    ap.add_argument("--topk-ent-floor", type=float, default=0.5,
                    help="entropy floor for top-k acceptance (Option B): accept only if the "
                         "true cb0 token is in top-k AND normalised entropy < this")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--save-json", type=Path, help="dump summary for plot_results.py")
    ap.add_argument("--save-arrays", type=Path, help="dump compact per-position arrays (.npz) for any later plot")
    args = ap.parse_args()

    import torch
    from duplex_spec.head import MultiStepTPPHead, MultiStepDepHead

    ck = torch.load(args.head, map_location=args.device)
    K = ck["horizon"]
    Head = MultiStepDepHead if ck.get("head_type") == "dep" else MultiStepTPPHead
    head = Head(hidden_dim=ck["hidden_dim"], n_channels=ck["n_channels"],
                n_codebooks=ck["n_codebooks"], codebook_size=2048, horizon=K)
    head.load_state_dict(ck["state_dict"]); head.to(args.device).eval()
    print(f"[head] type={ck.get('head_type','independent')} K={K} hidden={ck['hidden_dim']} "
          f"(val_loss at save={ck.get('val_loss','?')})")

    pairs = list(zip(args.feats, args.tokens))
    if args.pairs_dir:
        for npz in sorted(args.pairs_dir.glob("*.npz")):
            npy = npz.with_suffix(".npy")
            if npy.exists():
                pairs.append((npz, npy))
    if not pairs:
        sys.exit("No (feats, tokens) pairs.")

    m_list = [int(x) for x in args.stability_m.split(",")]
    rx_pairs = [tuple(int(v) for v in p.split(":")) for p in args.relaxed.split(",")] if args.relaxed else []
    floor = args.relaxed_ent_floor
    thresholds = [float(x) for x in args.thresholds.split(",")]
    logV = np.log(2048.0)

    # ---- pass 1: predictions + entropy + per-conversation stability commit lengths ----
    P, T, E, RK, E0, conv_sizes = [], [], [], [], [], []
    S = {m: [] for m in m_list}
    R = {wa: [] for wa in rx_pairs}
    with torch.no_grad():
        for fp, tp in pairs:
            d = np.load(fp); feats, frames = d["feats"], d["frames"]
            tokens = np.load(tp); C, Q, Tlen = tokens.shape
            rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + K < Tlen]
            if not rows:
                continue
            fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])
            pc, ec, tc, rk, e0 = [], [], [], [], []
            for s in range(0, len(ft_row), args.batch):
                br = ft_row[s:s + args.batch]; bf = fr_idx[s:s + args.batch]
                x = torch.from_numpy(feats[br].astype(np.float32)).to(args.device)
                lo = head(x)                                      # [b,K,C,Q,V]
                pc.append(lo.argmax(-1).cpu().numpy().astype(np.int16))
                p = torch.softmax(lo, dim=-1)
                ec.append(((-(p * (p + 1e-12).log()).sum(-1)).mean(dim=(2, 3)).cpu().numpy() / logV))
                # cb0 distribution + cb0-only entropy (top-k acceptance acts on cb0)
                p0 = p[:, :, :, 0, :]                             # [b,K,C,V]
                ent0 = (-(p0 * (p0 + 1e-12).log()).sum(-1)).cpu().numpy() / logV   # [b,K,C]
                tru = np.stack([tokens[:, :, f + 1:f + 1 + K] for f in bf]).transpose(0, 3, 1, 2)  # [b,K,C,Q]
                tru0 = tru[:, :, :, 0]                            # [b,K,C] true cb0 id
                # rank of the true cb0 token in each distribution: #tokens with prob>prob(true)
                tru0_t = torch.from_numpy(tru0.astype(np.int64)).to(args.device)   # [b,K,C]
                ptrue = torch.gather(p0, -1, tru0_t.unsqueeze(-1)).squeeze(-1)      # [b,K,C]
                rank0 = (p0 > ptrue.unsqueeze(-1)).sum(-1).cpu().numpy().astype(np.int32)  # [b,K,C]
                rk.append(rank0); e0.append(ent0)
                tc.append(tru.astype(np.int16))
            pred_c = np.concatenate(pc); ent_c = np.concatenate(ec); truth_c = np.concatenate(tc)
            rank_c = np.concatenate(rk); ent0_c = np.concatenate(e0)
            P.append(pred_c); E.append(ent_c); T.append(truth_c); RK.append(rank_c); E0.append(ent0_c)
            conv_sizes.append(len(pred_c))
            for m in m_list:
                S[m].append(stability_commit_lengths(pred_c, m))
            for (w, a) in rx_pairs:
                R[(w, a)].append(relaxed_commit_lengths(pred_c, w, a, ent=ent_c, ent_floor=floor))
    pred = np.concatenate(P); truth = np.concatenate(T); ent = np.concatenate(E)
    rank0 = np.concatenate(RK); ent0 = np.concatenate(E0)          # [N,K,C]
    for m in m_list:
        S[m] = np.concatenate(S[m])
    for wa in rx_pairs:
        R[wa] = np.concatenate(R[wa])
    N = len(pred)
    print(f"[data] {N} speculation points across {len(pairs)} conversation(s)\n")

    # top-k acceptance masks (Option B): accept horizon (n,k) iff, on BOTH channels, the true
    # cb0 token is within top-k AND the cb0 distribution is confident (norm-entropy < floor).
    topk_masks = {}
    for acc in [a for a in args.accept.split(",") if a.startswith("top")]:
        k = int(acc[3:])
        in_topk = (rank0 < k)                          # [N,K,C]
        confident = (ent0 < args.topk_ent_floor)       # [N,K,C]
        topk_masks[acc] = (in_topk & confident).all(axis=2)   # [N,K] both channels
    if topk_masks:
        print(f"[top-k] entropy floor={args.topk_ent_floor} on cb0; "
              f"modes={list(topk_masks)}")

    blob = {"meta": {"val_loss": float(ck.get("val_loss", 0) or 0), "n_points": int(N),
                     "n_convs": len(pairs)}, "results": {}}
    fa_by_accept = {}
    for accept in args.accept.split(","):
        fa = frame_acceptable_arr(pred, truth, accept, args.frac, topk_masks)
        lead = leading_run(fa)
        fa_by_accept[accept] = fa
        tag = f"{accept}" + (f"(frac={args.frac})" if accept == "frac" else "")
        print(f"================  acceptance = {tag}  ================")
        print(f"{'gate':>10} {'commit_prec':>11} {'rollback':>9} {'saved_ms':>8} {'committed':>10}")
        print("-" * 54)
        ent_rows, am_rows = [], []
        print("  ENTROPY (confidence, irrevocable):")
        for thr in thresholds:
            cl = leading_run(ent < thr)
            mt = metrics(cl, lead); print(row(f"thr={thr:.2f}", mt))
            ent_rows.append({"label": f"{thr:.2f}", "commit_prec": float(mt[0]),
                             "rollback": float(mt[1]), "saved_ms": float(mt[2]), "committed": float(mt[3])})
        print("  AMENDABLE (stability, convergence):")
        for m in m_list:
            mt = metrics(S[m], lead); print(row(f"m={m}", mt))
            am_rows.append({"label": f"m={m}", "commit_prec": float(mt[0]),
                            "rollback": float(mt[1]), "saved_ms": float(mt[2]), "committed": float(mt[3])})
        rx_rows = []
        if rx_pairs:
            ftag = f"/e<{floor}" if floor is not None else ""
            print("  RELAXED AMENDABLE (k-of-m vote" + (", +entropy floor" if floor is not None else "") + "):")
            for (w, a) in rx_pairs:
                mt = metrics(R[(w, a)], lead); print(row(f"w{w}:a{a}{ftag}", mt))
                rx_rows.append({"label": f"w{w}:a{a}{ftag}", "commit_prec": float(mt[0]),
                                "rollback": float(mt[1]), "saved_ms": float(mt[2]), "committed": float(mt[3])})
        blob["results"][accept] = {"entropy": ent_rows, "amendable": am_rows, "relaxed": rx_rows}
        print()
    if args.save_json:
        args.save_json.write_text(json.dumps(blob, indent=1))
        print(f"[json] {args.save_json}  (-> python scripts/plot_results.py --eval-json {args.save_json} --out figs/)")

    if args.save_arrays:
        out = {"entropy": ent.astype(np.float32),                       # [N,K]
               "m_list": np.array(m_list),
               "commit_lens": np.stack([S[m] for m in m_list]).astype(np.int16),  # [n_m, N]
               "relaxed_pairs": np.array(rx_pairs) if rx_pairs else np.zeros((0, 2), int),
               "relaxed_lens": (np.stack([R[wa] for wa in rx_pairs]).astype(np.int16) if rx_pairs else np.zeros((0, len(pred)), np.int16)),
               "conv_sizes": np.array(conv_sizes),
               "thresholds": np.array(thresholds),
               "val_loss": np.float32(blob["meta"]["val_loss"])}
        for accept, fa in fa_by_accept.items():
            out[f"fa_{accept}"] = fa.astype(np.uint8)                   # [N,K] per-horizon acceptable
        np.savez_compressed(args.save_arrays, **out)
        print(f"[npz ] {args.save_arrays}  (entropy, commit_lens, per-accept frame-acceptability, "
              f"conv boundaries -> recompute any plot offline)")

    print("Compare the two blocks at matched rollback: the amendable gate should\n"
          "hold higher commit_prec / saved_ms for the same rollback, because it\n"
          "refuses frames whose prediction is still being amended by new input —\n"
          "exactly the 'confidently wrong' frames entropy commits.")


if __name__ == "__main__":
    main()
