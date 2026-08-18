"""Three cheap diagnostics on CACHED v0 features --- run BEFORE building the
controller or the continuous-commit gate. Each can kill a week of work.

It reuses eval_speculative.py's exact prediction path (same head, same batching,
same vantage-point comparison as stability_commit_lengths) so the numbers align
with your real eval. The only addition: it keeps the full softmax and the cb0
embedding for each prediction, which the eval discards.

Diagnostics
-----------
(1) SILENCE STRATIFICATION
    What fraction of frames the amendable gate (m=3,4) COMMITS are
    silence-predicted-during-silence? If most commits are silence, the headline
    52%/1.6% is hiding no useful latency. Also splits commit precision by
    silence vs non-silence.

(2) AUTOCORRELATION OF COMMIT SUCCESS
    Does a rollback at frame t raise rollback probability at t+1..t+10 above the
    base rate? If flat -> difficulty is i.i.d. -> a history-based controller
    (AIMD / "elastic cord") has no signal to exploit -> don't build it.

(3) PHANTOM THRASH
    When cb0 argmax FLIPS between consecutive vantage points of the same frame,
    is the embedding L2 (Delta E) bimodal (a spike near 0 = acoustically
    interchangeable flips = the population a continuous gate would newly admit)?
    Sizes that population, split by entropy (low-ent = real; high-ent = garbage)
    and by silence, and checks how many such flips were actually CORRECT under
    cb0 exact-match anyway (i.e. how much is genuinely NEW coverage).

SILENCE TOKEN: not known a priori. We infer it as the single most frequent cb0
token id in the ground-truth stream (audio codecs spend most frames in
near-silence, so the modal cb0 token is silence with very high probability).
Override with --silence-id if you know it. The script prints the inferred id and
its corpus frequency so you can sanity-check.

Usage:
    PYTHONPATH=src python scripts/diagnose_commit.py \
        --head head_v0.pt --pairs-dir pairs_eval/ --device cuda
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--m", type=int, default=3, help="amendable m for the commit analysis")
    ap.add_argument("--silence-id", type=int, default=None,
                    help="cb0 token id for silence (default: inferred as modal cb0 token)")
    ap.add_argument("--max-lag", type=int, default=10)
    ap.add_argument("--save-json", type=Path, default=Path("diagnose_commit.json"))
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from duplex_spec.head import MultiStepTPPHead, MultiStepDepHead
    from duplex_spec.spec_eval import stability_commit_lengths

    ck = torch.load(args.head, map_location=args.device)
    K = ck["horizon"]
    Head = MultiStepDepHead if ck.get("head_type") == "dep" else MultiStepTPPHead
    head = Head(hidden_dim=ck["hidden_dim"], n_channels=ck["n_channels"],
                n_codebooks=ck["n_codebooks"], codebook_size=2048, horizon=K)
    head.load_state_dict(ck["state_dict"]); head.to(args.device).eval()
    C, Q = ck["n_channels"], ck["n_codebooks"]
    logV = np.log(2048.0)
    print(f"[head] K={K} C={C} Q={Q} hidden={ck['hidden_dim']}")

    # try to grab the cb0 embedding table from the head (for Delta E). If the head
    # has no output embedding we fall back to |id difference| as a weak proxy and say so.
    emb0 = None
    for name, p in head.named_parameters():
        # heuristic: an output projection for (channel 0, cb0) of shape [V, hidden] or [hidden, V]
        if "out" in name.lower() and p.dim() == 2 and 2048 in tuple(p.shape):
            emb0 = p.detach().float().cpu().numpy()
            if emb0.shape[0] != 2048:
                emb0 = emb0.T                                  # -> [V, hidden]
            print(f"[emb] using '{name}' {emb0.shape} as cb0 code geometry for Delta E")
            break
    if emb0 is None:
        print("[emb] WARNING: no cb0 embedding table found on the head; Delta E will use "
              "|id difference| as a weak proxy (phantom-thrash geometry unreliable).")

    pairs = []
    for npz in sorted(args.pairs_dir.glob("*.npz")):
        npy = npz.with_suffix(".npy")
        if npy.exists():
            pairs.append((npz, npy))
    if not pairs:
        sys.exit("No (feats, tokens) pairs found.")

    # ---- pass 1: per-conversation predictions, softmax(cb0), truth, entropy ----
    # keep per-conversation so vantage-point offsets never cross conversation boundaries.
    convs = []            # each: dict(pred[N,K,C,Q], truth[N,K,C,Q], ent[N,K], p0[N,K,C,V])
    cb0_truth_all = []
    with torch.no_grad():
        for fp, tp in pairs:
            d = np.load(fp); feats, frames = d["feats"], d["frames"]
            tokens = np.load(tp); _C, _Q, Tlen = tokens.shape
            rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + K < Tlen]
            if not rows:
                continue
            fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])
            pc, ec, tc, p0c = [], [], [], []
            for s in range(0, len(ft_row), args.batch):
                br = ft_row[s:s + args.batch]; bf = fr_idx[s:s + args.batch]
                x = torch.from_numpy(feats[br].astype(np.float32)).to(args.device)
                lo = head(x)                                   # [b,K,C,Q,V]
                pc.append(lo.argmax(-1).cpu().numpy().astype(np.int16))
                p = torch.softmax(lo, dim=-1)
                ec.append(((-(p * (p + 1e-12).log()).sum(-1)).mean(dim=(2, 3)).cpu().numpy() / logV))
                p0c.append(p[:, :, :, 0, :].cpu().numpy().astype(np.float32))   # cb0 dist [b,K,C,V]
                tc.append(np.stack([tokens[:, :, f + 1:f + 1 + K] for f in bf]
                                   ).transpose(0, 3, 1, 2).astype(np.int16))
            convs.append({"pred": np.concatenate(pc), "truth": np.concatenate(tc),
                          "ent": np.concatenate(ec), "p0": np.concatenate(p0c)})
            cb0_truth_all.append(tokens[:, 0, :].reshape(-1))   # both channels' cb0 over time
    if not convs:
        sys.exit("No usable speculation points.")

    # ---- infer silence token ----
    cb0_truth_all = np.concatenate(cb0_truth_all)
    if args.silence_id is None:
        vals, counts = np.unique(cb0_truth_all, return_counts=True)
        sil = int(vals[counts.argmax()]); frac = counts.max() / counts.sum()
        print(f"[silence] inferred cb0 silence id = {sil} "
              f"({frac:.1%} of all cb0 frames). Override with --silence-id if wrong.")
    else:
        sil = args.silence_id
        print(f"[silence] using provided cb0 silence id = {sil}")

    N_total = sum(len(c["pred"]) for c in convs)
    print(f"[data] {N_total} speculation points across {len(convs)} conversation(s)\n")

    report = {}

    # ======================================================================
    # (1) SILENCE STRATIFICATION of amendable commits
    # ======================================================================
    m = args.m
    tot_commit = 0; sil_commit = 0
    prec_num_sil = prec_den_sil = prec_num_ns = prec_den_ns = 0
    for c in convs:
        pred, truth = c["pred"], c["truth"]
        clen = stability_commit_lengths(pred, m)               # [N] leading committed horizons
        for i in range(len(pred)):
            n = int(clen[i])
            for k in range(n):
                tot_commit += 1
                pred_is_sil = np.all(pred[i, k, :, 0] == sil)
                correct = np.all(pred[i, k, :, 0] == truth[i, k, :, 0])   # cb0 accept
                if pred_is_sil:
                    sil_commit += 1
                    prec_den_sil += 1; prec_num_sil += int(correct)
                else:
                    prec_den_ns += 1; prec_num_ns += int(correct)
    if tot_commit:
        print("=== (1) SILENCE STRATIFICATION of amendable commits (m=%d, cb0) ===" % m)
        print(f"  committed frames total      : {tot_commit}")
        print(f"  committed that are SILENCE   : {sil_commit} ({sil_commit/tot_commit:.1%})")
        ps = prec_num_sil/prec_den_sil if prec_den_sil else float('nan')
        pn = prec_num_ns/prec_den_ns if prec_den_ns else float('nan')
        print(f"  commit precision | silence   : {ps:.1%} (n={prec_den_sil})")
        print(f"  commit precision | non-silence: {pn:.1%} (n={prec_den_ns})")
        print(f"  --> if silence share is high AND non-silence precision is low, the")
        print(f"      headline number is mostly 'predicting silence during silence'.\n")
        report["silence"] = {"m": m, "committed": tot_commit,
                             "silence_committed": sil_commit,
                             "silence_frac": sil_commit/tot_commit,
                             "prec_silence": ps, "prec_nonsilence": pn,
                             "n_silence": prec_den_sil, "n_nonsilence": prec_den_ns}

    # ======================================================================
    # (2) AUTOCORRELATION of commit success (per conversation, horizon k=1)
    # ======================================================================
    # define a per-frame "rollback event": at horizon k=1, the committed (here:
    # always-attempted) cb0 prediction was wrong. We use k=1 as the cleanest
    # per-frame signal. Base rate vs rate conditioned on a rollback `lag` frames ago.
    base_hits = base_tot = 0
    cond_hits = np.zeros(args.max_lag + 1); cond_tot = np.zeros(args.max_lag + 1)
    for c in convs:
        pred, truth = c["pred"], c["truth"]
        wrong = ~np.all(pred[:, 0, :, 0] == truth[:, 0, :, 0], axis=1)   # [N] bool, k=1
        base_hits += int(wrong.sum()); base_tot += len(wrong)
        for lag in range(1, args.max_lag + 1):
            if len(wrong) <= lag:
                continue
            prev = wrong[:-lag]; curr = wrong[lag:]
            sel = curr[prev]                                   # current-wrong given prev-wrong
            cond_hits[lag] += int(sel.sum()); cond_tot[lag] += int(prev.sum())
    base = base_hits / max(base_tot, 1)
    print("=== (2) AUTOCORRELATION of rollback (k=1 cb0 error) ===")
    print(f"  base rollback rate: {base:.3f}")
    print(f"  {'lag':>4} {'P(wrong|wrong lag ago)':>24} {'lift vs base':>14}")
    ac = {}
    for lag in range(1, args.max_lag + 1):
        if cond_tot[lag]:
            p = cond_hits[lag] / cond_tot[lag]
            print(f"  {lag:>4} {p:>24.3f} {p/base if base else float('nan'):>13.2f}x")
            ac[lag] = p
    print("  --> lift >> 1 at small lags = clustered difficulty = a controller has signal.")
    print("      lift ~ 1 everywhere = i.i.d. = AIMD/elastic-cord won't help.\n")
    report["autocorr"] = {"base": base, "cond": ac}

    # ---- (2b) per-FRAME vs per-SPECULATION error, to show how harsh the binary flag is ----
    # per-frame: fraction of individual committed frames that are wrong (cb0), at m.
    # per-spec : fraction of speculations with >=1 wrong committed frame (the rollback flag).
    from duplex_spec.spec_eval import stability_commit_lengths as _scl
    pf_wrong = pf_tot = 0; ps_roll = ps_tot = 0
    for c in convs:
        pred, truth = c["pred"], c["truth"]
        clen = _scl(pred, args.m)
        for i in range(len(pred)):
            n = int(clen[i])
            if n == 0:
                continue
            ps_tot += 1
            spec_has_wrong = False
            for k in range(n):
                ok = np.all(pred[i, k, :, 0] == truth[i, k, :, 0])
                pf_tot += 1; pf_wrong += int(not ok)
                if not ok:
                    spec_has_wrong = True
            ps_roll += int(spec_has_wrong)
    if pf_tot and ps_tot:
        pf = pf_wrong / pf_tot; ps = ps_roll / ps_tot
        print("=== (2b) PER-FRAME vs PER-SPECULATION error (m=%d, cb0, committed frames only) ===" % args.m)
        print(f"  per-FRAME error       : {pf:.1%}  (wrong committed frames / all committed frames)")
        print(f"  per-SPECULATION roll  : {ps:.1%}  (specs with >=1 wrong / all specs that committed)")
        print(f"  --> a large gap means the binary rollback flag inflates failure: many")
        print(f"      speculations are mostly-right but flagged for a single late miss.\n")
        report["binary_gap"] = {"per_frame_error": pf, "per_spec_rollback": ps}

    # ======================================================================
    # (3) PHANTOM THRASH: cb0 argmax flips vs Delta E / JS / entropy / silence
    # ======================================================================
    def js(p, q):
        m_ = 0.5 * (p + q)
        def kl(a, b):
            a = np.clip(a, 1e-8, 1.0); b = np.clip(b, 1e-8, 1.0)
            return np.sum(a * np.log(a / b), axis=-1)
        return 0.5 * kl(p, m_) + 0.5 * kl(q, m_)

    dE_flip = []; js_flip = []; ent_flip = []; sil_flip = []; correct_flip = []
    truerank_flip = []            # rank of the TRUE cb0 token in the current (stable) dist
    n_pairs_compared = 0
    for c in convs:
        pred, truth, ent, p0 = c["pred"], c["truth"], c["ent"], c["p0"]
        Ncc = len(pred)
        # compare vantage points of the SAME absolute frame: horizon k at position i
        # vs horizon k+1 at position i-1 (exactly the stability comparison, j=1).
        for k in range(K - 1):
            a_pos = np.arange(1, Ncc)                       # position i
            # current prediction (more-informed): position i, horizon k
            cur_id = pred[a_pos, k, :, 0]                   # [., C]
            prev_id = pred[a_pos - 1, k + 1, :, 0]          # [., C] same frame, 1 step earlier
            cur_p = p0[a_pos, k, :, :]                      # [., C, V]
            prev_p = p0[a_pos - 1, k + 1, :, :]
            tru_id = truth[a_pos, k, :, 0]                  # [., C]
            flip = np.any(cur_id != prev_id, axis=1)        # argmax flipped on some channel
            if not flip.any():
                continue
            fi = np.where(flip)[0]
            n_pairs_compared += len(a_pos)
            # Delta E (channel-averaged), JS (channel-averaged), current entropy at (i,k)
            for idx in fi:
                cc = cur_id[idx]; pp = prev_id[idx]
                if emb0 is not None:
                    de = np.mean([np.linalg.norm(emb0[cc[ch]] - emb0[pp[ch]]) for ch in range(C)])
                else:
                    de = float(np.mean(np.abs(cc.astype(int) - pp.astype(int))))
                jsd = float(np.mean(js(cur_p[idx], prev_p[idx])))
                # rank of true token in the current distribution (0 = top-1), per channel then avg
                ranks = []
                for ch in range(C):
                    dist = cur_p[idx, ch]; tid = int(tru_id[idx, ch])
                    ranks.append(int((dist > dist[tid]).sum()))
                truerank_flip.append(float(np.mean(ranks)))
                dE_flip.append(de); js_flip.append(jsd)
                ent_flip.append(float(ent[a_pos[idx], k]))
                sil_flip.append(bool(np.all(cur_id[idx] == sil)))
                correct_flip.append(bool(np.all(cur_id[idx] == tru_id[idx])))
    dE_flip = np.array(dE_flip); js_flip = np.array(js_flip)
    ent_flip = np.array(ent_flip); sil_flip = np.array(sil_flip); correct_flip = np.array(correct_flip)
    nflip = len(dE_flip)
    print("=== (3) PHANTOM THRASH (cb0 argmax flips across consecutive vantage points) ===")
    print(f"  total flips analysed: {nflip}")
    if nflip:
        # "phantom" = low JS (distributions barely moved) AND low entropy (confident)
        js_med = np.median(js_flip)
        low_js = js_flip < js_med
        low_ent = ent_flip < 0.5
        phantom = low_js & low_ent
        print(f"  Delta E: min={dE_flip.min():.3f} median={np.median(dE_flip):.3f} "
              f"max={dE_flip.max():.3f}  (bimodal w/ a spike near 0 => real phantom thrash)")
        print(f"  JS between the two distributions: median={js_med:.4f}")
        print(f"  flips with low-JS & low-entropy (candidate 'phantom'): "
              f"{phantom.sum()} ({phantom.mean():.1%} of flips)")
        if phantom.sum():
            print(f"     of those, silence      : {sil_flip[phantom].mean():.1%}")
            print(f"     of those, already-correct under cb0 exact-match: "
                  f"{correct_flip[phantom].mean():.1%}")
            newcov = phantom & (~correct_flip)
            print(f"     --> NEW committable & not already correct: {newcov.sum()} "
                  f"({newcov.mean():.1%} of flips); non-silence of those: "
                  f"{(newcov & ~sil_flip).sum()}")
            # Delta E distribution of the PHANTOM subset only -> where to set a continuous threshold
            dP = dE_flip[phantom]
            if len(dP):
                qs = np.percentile(dP, [10, 25, 50, 75, 90])
                print(f"     phantom Delta E pctiles [10/25/50/75/90]: "
                      f"{qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.2f}/{qs[4]:.2f}")
                # how many phantom flips are ALSO acoustically tiny (interchangeable):
                for thr in (1.0, 2.0, 5.0):
                    frac = (dP < thr).mean()
                    print(f"     phantom flips with Delta E < {thr:>4.1f}: {frac:.1%}")
                print(f"     --> a continuous gate keyed on Delta E would commit the low-Delta E")
                print(f"         tail here; these pctiles suggest where the threshold sits.")
                report.setdefault("phantom", {})["dE_pctiles"] = [float(x) for x in qs]
            # DECISIVE: for phantom flips, is the TRUE token in the high-prob region of the
            # stable distribution even though the argmax flipped? If yes, a distribution-based
            # accept/commit rule rescues them; if no, the stability is illusory.
            tr = np.array(truerank_flip)[phantom]
            if len(tr):
                print(f"     phantom TRUE-token rank in stable dist: "
                      f"median={np.median(tr):.0f}  mean={tr.mean():.1f}")
                for kk in (1, 5, 10, 50):
                    print(f"       true token in top-{kk:<3d}: {(tr < kk).mean():.1%}")
                print(f"     --> high top-5/top-10 => the stable distribution KNOWS the answer,")
                print(f"         argmax just misreads it: a JS/distribution gate is worth building.")
                print(f"         near-zero => stability is illusory, continuous gate won't help.")
                report["phantom"]["true_rank_median"] = float(np.median(tr))
                report["phantom"]["true_in_top5"] = float((tr < 5).mean())
                report["phantom"]["true_in_top10"] = float((tr < 10).mean())
        print("  --> the last number (new, non-silence, not-already-correct) is the real")
        print("      ceiling a continuous gate could add. If tiny, don't build it.\n")
        report["phantom"] = {"flips": int(nflip),
                             "dE_min": float(dE_flip.min()), "dE_med": float(np.median(dE_flip)),
                             "dE_max": float(dE_flip.max()), "js_med": float(js_med),
                             "phantom_frac": float(phantom.mean()),
                             "phantom_silence_frac": float(sil_flip[phantom].mean()) if phantom.sum() else 0.0,
                             "phantom_already_correct": float(correct_flip[phantom].mean()) if phantom.sum() else 0.0}

    import json
    args.save_json.write_text(json.dumps(report, indent=2))
    print(f"[json] {args.save_json}")


if __name__ == "__main__":
    main()
