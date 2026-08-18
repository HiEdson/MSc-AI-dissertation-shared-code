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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from duplex_spec.spec_eval import stability_commit_lengths  # noqa: E402

MS_PER_FRAME = 80.0


def leading_run(b: np.ndarray) -> np.ndarray:
    """[N,K] bool -> [N] count of leading True per row."""
    return np.cumprod(b, axis=1).sum(axis=1).astype(int)


def leading_acceptable(pred, truth, accept, frac):
    if accept == "cb0":
        fa = (pred[:, :, :, 0] == truth[:, :, :, 0]).all(axis=2)        # [N,K]
    elif accept == "exact":
        fa = (pred == truth).all(axis=(2, 3))
    else:  # frac
        fa = (pred == truth).mean(axis=(2, 3)) >= frac
    return leading_run(fa)


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
    ap.add_argument("--accept", default="cb0,exact")
    ap.add_argument("--frac", type=float, default=0.5)
    args = ap.parse_args()

    import torch
    from duplex_spec.head import MultiStepTPPHead

    ck = torch.load(args.head, map_location=args.device)
    K = ck["horizon"]
    head = MultiStepTPPHead(hidden_dim=ck["hidden_dim"], n_channels=ck["n_channels"],
                            n_codebooks=ck["n_codebooks"], codebook_size=2048, horizon=K)
    head.load_state_dict(ck["state_dict"]); head.to(args.device).eval()
    print(f"[head] K={K} hidden={ck['hidden_dim']} (val_loss at save={ck.get('val_loss','?')})")

    pairs = list(zip(args.feats, args.tokens))
    if args.pairs_dir:
        for npz in sorted(args.pairs_dir.glob("*.npz")):
            npy = npz.with_suffix(".npy")
            if npy.exists():
                pairs.append((npz, npy))
    if not pairs:
        sys.exit("No (feats, tokens) pairs.")

    m_list = [int(x) for x in args.stability_m.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    logV = np.log(2048.0)

    # ---- pass 1: predictions + entropy + per-conversation stability commit lengths ----
    P, T, E = [], [], []
    S = {m: [] for m in m_list}
    with torch.no_grad():
        for fp, tp in pairs:
            d = np.load(fp); feats, frames = d["feats"], d["frames"]
            tokens = np.load(tp); C, Q, Tlen = tokens.shape
            rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + K < Tlen]
            if not rows:
                continue
            fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])
            pc, ec, tc = [], [], []
            for s in range(0, len(ft_row), args.batch):
                br = ft_row[s:s + args.batch]; bf = fr_idx[s:s + args.batch]
                x = torch.from_numpy(feats[br].astype(np.float32)).to(args.device)
                lo = head(x)                                      # [b,K,C,Q,V]
                pc.append(lo.argmax(-1).cpu().numpy().astype(np.int16))
                p = torch.softmax(lo, dim=-1)
                ec.append(((-(p * (p + 1e-12).log()).sum(-1)).mean(dim=(2, 3)).cpu().numpy() / logV))
                tc.append(np.stack([tokens[:, :, f + 1:f + 1 + K] for f in bf]).transpose(0, 3, 1, 2).astype(np.int16))
            pred_c = np.concatenate(pc); ent_c = np.concatenate(ec); truth_c = np.concatenate(tc)
            P.append(pred_c); E.append(ent_c); T.append(truth_c)
            for m in m_list:
                S[m].append(stability_commit_lengths(pred_c, m))
    pred = np.concatenate(P); truth = np.concatenate(T); ent = np.concatenate(E)
    for m in m_list:
        S[m] = np.concatenate(S[m])
    N = len(pred)
    print(f"[data] {N} speculation points across {len(pairs)} conversation(s)\n")

    for accept in args.accept.split(","):
        lead = leading_acceptable(pred, truth, accept, args.frac)
        tag = f"{accept}" + (f"(frac={args.frac})" if accept == "frac" else "")
        print(f"================  acceptance = {tag}  ================")
        print(f"{'gate':>10} {'commit_prec':>11} {'rollback':>9} {'saved_ms':>8} {'committed':>10}")
        print("-" * 54)
        print("  ENTROPY (confidence, irrevocable):")
        for thr in thresholds:
            cl = leading_run(ent < thr)
            print(row(f"thr={thr:.2f}", metrics(cl, lead)))
        print("  AMENDABLE (stability, convergence):")
        for m in m_list:
            print(row(f"m={m}", metrics(S[m], lead)))
        print()

    print("Compare the two blocks at matched rollback: the amendable gate should\n"
          "hold higher commit_prec / saved_ms for the same rollback, because it\n"
          "refuses frames whose prediction is still being amended by new input —\n"
          "exactly the 'confidently wrong' frames entropy commits.")


if __name__ == "__main__":
    main()
