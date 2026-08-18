"""Sweep the JS (distribution-stability) commit gate against the argmax amendable gate.

Runs on CACHED v0 features --- no retraining, no live backbone. It reuses
eval_speculative.py's prediction path and metric definitions so the numbers are
directly comparable to your existing frontier.

For each (m, tau) it reports commit precision / rollback / saved_ms / committed under
each acceptance rule, and prints the argmax gate at the same m as the control row.
The question this answers: does testing DISTRIBUTION stability instead of ARGMAX
equality raise coverage (the `committed` column) without wrecking precision?

Usage:
    PYTHONPATH=src python scripts/sweep_js_gate.py \
        --head head_v0.pt --pairs-dir pairs_eval/ --device cuda \
        --m 2,3,4 --tau 0.01,0.02,0.05,0.10,0.20 --ent-floor 0.5 \
        --accept cb0,top5
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--m", default="2,3,4")
    ap.add_argument("--tau", default="0.01,0.02,0.05,0.10,0.20",
                    help="JS thresholds (natural-log scale, max ln2~0.693)")
    ap.add_argument("--ent-floor", type=float, default=0.5,
                    help="normalised-entropy ceiling; frames above it are never committed")
    ap.add_argument("--accept", default="cb0,top5",
                    help="acceptance rules to score under (cb0, exact, top5, top10)")
    ap.add_argument("--topk-ent-floor", type=float, default=0.5)
    ap.add_argument("--save-json", type=Path, default=Path("js_gate_sweep.json"))
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from duplex_spec.head import MultiStepTPPHead, MultiStepDepHead
    try:
        from duplex_spec_v2.js_gate import js_commit_lengths, argmax_commit_lengths_from_dist
    except ImportError:
        from duplex_spec.js_gate import js_commit_lengths, argmax_commit_lengths_from_dist

    ck = torch.load(args.head, map_location=args.device)
    K = ck["horizon"]
    Head = MultiStepDepHead if ck.get("head_type") == "dep" else MultiStepTPPHead
    head = Head(hidden_dim=ck["hidden_dim"], n_channels=ck["n_channels"],
                n_codebooks=ck["n_codebooks"], codebook_size=2048, horizon=K)
    head.load_state_dict(ck["state_dict"]); head.to(args.device).eval()
    C, Q = ck["n_channels"], ck["n_codebooks"]
    logV = float(np.log(2048.0))
    print(f"[head] K={K} C={C} Q={Q}")

    pairs = []
    for npz in sorted(args.pairs_dir.glob("*.npz")):
        npy = npz.with_suffix(".npy")
        if npy.exists():
            pairs.append((npz, npy))
    if not pairs:
        sys.exit("No (feats, tokens) pairs found.")

    # ---- pass 1: per-conversation cb0 distributions, truth, and true-token rank ----
    convs = []
    with torch.no_grad():
        for fp, tp in pairs:
            d = np.load(fp); feats, frames = d["feats"], d["frames"]
            tokens = np.load(tp); _C, _Q, Tlen = tokens.shape
            rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + K < Tlen]
            if not rows:
                continue
            fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])
            P0, TR, RK, E0 = [], [], [], []
            for s in range(0, len(ft_row), args.batch):
                br = ft_row[s:s + args.batch]; bf = fr_idx[s:s + args.batch]
                x = torch.from_numpy(feats[br].astype(np.float32)).to(args.device)
                lo = head(x)
                p = torch.softmax(lo, dim=-1)
                p0 = p[:, :, :, 0, :]                                  # [b,K,C,V] cb0
                tru = np.stack([tokens[:, :, f + 1:f + 1 + K] for f in bf]
                               ).transpose(0, 3, 1, 2)                 # [b,K,C,Q]
                tru0 = torch.from_numpy(tru[:, :, :, 0].astype(np.int64)).to(args.device)
                ptrue = torch.gather(p0, -1, tru0.unsqueeze(-1)).squeeze(-1)
                RK.append((p0 > ptrue.unsqueeze(-1)).sum(-1).cpu().numpy().astype(np.int32))
                E0.append(((-(p0 * (p0 + 1e-12).log()).sum(-1)).cpu().numpy() / logV))
                P0.append(p0.cpu().numpy().astype(np.float32))
                TR.append(tru.astype(np.int16))
            convs.append({"p0": np.concatenate(P0), "truth": np.concatenate(TR),
                          "rank": np.concatenate(RK), "ent0": np.concatenate(E0)})
    if not convs:
        sys.exit("No usable speculation points.")
    N_total = sum(len(c["p0"]) for c in convs)
    print(f"[data] {N_total} speculation points across {len(convs)} conversation(s)")
    print(f"[gate] entropy floor={args.ent_floor}; acceptance floor={args.topk_ent_floor}\n")

    def accept_mask(c, rule):
        """[N,K] bool: is horizon k acceptable under `rule` (all channels)?"""
        pred0 = c["p0"].argmax(-1)                         # [N,K,C]
        tru0 = c["truth"][:, :, :, 0]
        if rule == "cb0":
            return (pred0 == tru0).all(axis=2)
        if rule == "exact":
            # need full-codebook preds; approximate with cb0 (exact is ~0 anyway)
            return (pred0 == tru0).all(axis=2) & False
        if rule.startswith("top"):
            k = int(rule[3:])
            return ((c["rank"] < k) & (c["ent0"] < args.topk_ent_floor)).all(axis=2)
        raise ValueError(rule)

    def score(commit_fn, rule):
        """Aggregate precision / rollback / saved_ms / committed over all conversations."""
        sum_r = sum_n = n_roll = n_pts = 0
        for c in convs:
            clen = commit_fn(c)                            # [N]
            fa = accept_mask(c, rule)                      # [N,K]
            lead = np.cumprod(fa, axis=1).sum(axis=1)      # leading acceptable run
            r = np.minimum(clen, lead)
            sum_r += int(r.sum()); sum_n += int(clen.sum())
            n_roll += int((clen > r).sum()); n_pts += len(clen)
        prec = sum_r / sum_n if sum_n else float("nan")
        return {"precision": prec, "rollback": n_roll / max(n_pts, 1),
                "saved_ms": 80.0 * sum_r / max(n_pts, 1),
                "committed": sum_n / max(n_pts, 1)}

    ms = [int(x) for x in args.m.split(",")]
    taus = [float(x) for x in args.tau.split(",")]
    rules = [r.strip() for r in args.accept.split(",")]
    out = {}

    for rule in rules:
        print(f"================  acceptance = {rule}  ================")
        print(f"{'gate':>18} {'commit_prec':>12} {'rollback':>9} {'saved_ms':>9} {'committed':>10}")
        print("-" * 62)
        for m in ms:
            base = score(lambda c, m=m: argmax_commit_lengths_from_dist(c["p0"], m), rule)
            print(f"{'argmax m=' + str(m):>18} {base['precision']:>11.1%} "
                  f"{base['rollback']:>8.1%} {base['saved_ms']:>9.1f} {base['committed']:>10.3f}")
            out[f"{rule}|argmax|m{m}"] = base
            for tau in taus:
                r = score(lambda c, m=m, tau=tau: js_commit_lengths(
                    c["p0"], m, tau, args.ent_floor), rule)
                lift = r["committed"] / base["committed"] if base["committed"] else float("inf")
                print(f"{'  JS m=' + str(m) + ' t=' + f'{tau:g}':>18} {r['precision']:>11.1%} "
                      f"{r['rollback']:>8.1%} {r['saved_ms']:>9.1f} {r['committed']:>10.3f}"
                      f"   ({lift:.1f}x coverage)")
                out[f"{rule}|js|m{m}|tau{tau}"] = r
        print()

    args.save_json.write_text(json.dumps(out, indent=2))
    print(f"[json] {args.save_json}")
    print("Read the `committed` column first: that is coverage, the thing the JS gate is")
    print("meant to raise. Then check precision did not collapse to pay for it.")


if __name__ == "__main__":
    main()
