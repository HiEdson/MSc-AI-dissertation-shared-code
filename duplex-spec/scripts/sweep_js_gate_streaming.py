"""JS (distribution-stability) commit gate sweep --- STREAMING version.

Same experiment as sweep_js_gate.py, but processes one conversation at a time and keeps
only running scalar totals, so memory stays bounded and it runs over the full held-out
set instead of OOM-killing on the third conversation.

Memory note: one conversation's cb0 distributions are [N, K, C, V] float32 --- about
1.3 GB for a 20k-frame conversation at K=4, C=2, V=2048. That is held for one
conversation only, then freed. Peak RSS stays ~2 GB regardless of how many
conversations you evaluate.

The gate: commit horizon k iff, on every channel and across the last m vantage points,
JS(dist_now, dist_earlier) < tau AND normalised entropy < ent_floor. Argmax equality is
the degenerate case where the peak happens to hold, so this generalises the amendable
criterion. Argmax rows are printed as the control at each m.

Usage:
    PYTHONPATH=src python scripts/sweep_js_gate_streaming.py \
        --head head_v0.pt --pairs-dir pairs_eval/ --device cuda \
        --m 3,4 --tau 0.1,0.2,0.3,0.4,0.5,0.6,0.65 --ent-floor 0.5 \
        --accept cb0,top5,top10
"""
from __future__ import annotations
import argparse, sys, json, gc
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--m", default="3,4")
    ap.add_argument("--tau", default="0.1,0.2,0.3,0.4,0.5,0.6,0.65",
                    help="JS thresholds (natural log, max ln2~0.693)")
    ap.add_argument("--ent-floor", type=float, default=0.5)
    ap.add_argument("--accept", default="cb0,top5")
    ap.add_argument("--topk-ent-floor", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per conversation (memory safety valve)")
    ap.add_argument("--save-json", type=Path, default=Path("js_gate_sweep_full.json"))
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
    logV = float(np.log(2048.0))

    pairs = []
    for npz in sorted(args.pairs_dir.glob("*.npz")):
        npy = npz.with_suffix(".npy")
        if npy.exists():
            pairs.append((npz, npy))
    if not pairs:
        sys.exit("No (feats, tokens) pairs found.")

    ms = [int(x) for x in args.m.split(",")]
    taus = [float(x) for x in args.tau.split(",")]
    rules = [r.strip() for r in args.accept.split(",")]

    # configs: (rule, kind, m, tau) -> running totals
    def key(rule, kind, m, tau=None):
        return f"{rule}|{kind}|m{m}" + (f"|tau{tau}" if tau is not None else "")
    tot = {}
    for rule in rules:
        for m in ms:
            tot[key(rule, "argmax", m)] = dict(r=0, n=0, roll=0, pts=0)
            for tau in taus:
                tot[key(rule, "js", m, tau)] = dict(r=0, n=0, roll=0, pts=0)

    print(f"[head] K={K}  |  {len(pairs)} conversation(s)  |  streaming, bounded memory")
    print(f"[gate] commit entropy floor={args.ent_floor}; top-k acceptance floor={args.topk_ent_floor}\n")

    for ci, (fp, tp) in enumerate(pairs, 1):
        d = np.load(fp); feats, frames = d["feats"], d["frames"]
        tokens = np.load(tp); Tlen = tokens.shape[2]
        rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + K < Tlen]
        if args.max_frames:
            rows = rows[: args.max_frames]
        if not rows:
            continue
        fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])

        P0, TR, RK, E0 = [], [], [], []
        with torch.no_grad():
            for s in range(0, len(ft_row), args.batch):
                br = ft_row[s:s + args.batch]; bf = fr_idx[s:s + args.batch]
                x = torch.from_numpy(feats[br].astype(np.float32)).to(args.device)
                p = torch.softmax(head(x), dim=-1)
                p0 = p[:, :, :, 0, :]                                   # [b,K,C,V]
                tru = np.stack([tokens[:, :, f + 1:f + 1 + K] for f in bf]
                               ).transpose(0, 3, 1, 2)                  # [b,K,C,Q]
                tru0 = torch.from_numpy(tru[:, :, :, 0].astype(np.int64)).to(args.device)
                ptrue = torch.gather(p0, -1, tru0.unsqueeze(-1)).squeeze(-1)
                RK.append((p0 > ptrue.unsqueeze(-1)).sum(-1).cpu().numpy().astype(np.int32))
                E0.append(((-(p0 * (p0 + 1e-12).log()).sum(-1)).cpu().numpy() / logV))
                P0.append(p0.cpu().numpy().astype(np.float32))
                TR.append(tru[:, :, :, 0].astype(np.int16))             # cb0 truth only
                del p, p0, x
        p0 = np.concatenate(P0); truth0 = np.concatenate(TR)
        rank = np.concatenate(RK); ent0 = np.concatenate(E0)
        del P0, TR, RK, E0

        pred0 = p0.argmax(-1)                                           # [N,K,C]
        masks = {}
        for rule in rules:
            if rule == "cb0":
                masks[rule] = (pred0 == truth0).all(axis=2)
            elif rule.startswith("top"):
                kk = int(rule[3:])
                masks[rule] = ((rank < kk) & (ent0 < args.topk_ent_floor)).all(axis=2)
            else:
                raise ValueError(f"unsupported acceptance rule: {rule}")
        leads = {r: np.cumprod(mk, axis=1).sum(axis=1) for r, mk in masks.items()}

        def accumulate(k, clen):
            for rule in rules:
                lead = leads[rule]
                r = np.minimum(clen, lead)
                t = tot[k.replace("RULE", rule)]
                t["r"] += int(r.sum()); t["n"] += int(clen.sum())
                t["roll"] += int((clen > r).sum()); t["pts"] += len(clen)

        for m in ms:
            accumulate(key("RULE", "argmax", m), argmax_commit_lengths_from_dist(p0, m))
            for tau in taus:
                accumulate(key("RULE", "js", m, tau),
                           js_commit_lengths(p0, m, tau, args.ent_floor))

        print(f"  [{ci}/{len(pairs)}] {fp.stem}: {len(p0)} points")
        del p0, truth0, rank, ent0, pred0, masks, leads
        gc.collect()

    def fmt(t):
        prec = t["r"] / t["n"] if t["n"] else float("nan")
        return (prec, t["roll"] / max(t["pts"], 1),
                80.0 * t["r"] / max(t["pts"], 1), t["n"] / max(t["pts"], 1))

    out = {}
    for rule in rules:
        print(f"\n================  acceptance = {rule}  ================")
        print(f"{'gate':>18} {'commit_prec':>12} {'rollback':>9} {'saved_ms':>9} {'committed':>10}")
        print("-" * 62)
        for m in ms:
            bp, br_, bs, bc = fmt(tot[key(rule, "argmax", m)])
            print(f"{'argmax m=' + str(m):>18} {bp:>11.1%} {br_:>8.1%} {bs:>9.1f} {bc:>10.3f}")
            out[key(rule, "argmax", m)] = dict(precision=bp, rollback=br_, saved_ms=bs, committed=bc)
            best = None
            for tau in taus:
                pr, ro, sv, co = fmt(tot[key(rule, "js", m, tau)])
                lift = co / bc if bc else float("inf")
                star = ""
                if best is None or sv > best[1]:
                    best = (tau, sv)
                print(f"{'  JS m=' + str(m) + ' t=' + f'{tau:g}':>18} {pr:>11.1%} {ro:>8.1%} "
                      f"{sv:>9.1f} {co:>10.3f}   ({lift:.1f}x coverage){star}")
                out[key(rule, "js", m, tau)] = dict(precision=pr, rollback=ro,
                                                    saved_ms=sv, committed=co)
            if best:
                print(f"{'':>18}  peak saved_ms at tau={best[0]:g} ({best[1]:.1f} ms)")

    args.save_json.write_text(json.dumps(out, indent=2))
    print(f"\n[json] {args.save_json}")
    print("Read `committed` (coverage) first, then check what precision it cost.")
    print("tau where JS beats argmax on BOTH precision and coverage = strict dominance;")
    print("above that, it becomes an ordinary trade you can dial to a glitch budget.")


if __name__ == "__main__":
    main()
