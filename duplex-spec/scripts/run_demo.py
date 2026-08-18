"""Unified demo harness --- runs every trigger through ONE continuous gate/release loop and
reports them in a single table, so all your contributions are compared on the same footing.

Pipeline per conversation:
  1. Precompute per-frame FrameState from the frozen head (cb0 dist now + one tick ago,
     entropy, hidden, activity). One pass; every trigger reads identical inputs.
  2. For each trigger (calibrated to the rollback budget), run the CONTINUOUS loop:
       - Moshi starts gated (silent/listening).
       - each frame, trigger.fire(state) decides release; while released Moshi speaks,
         and the trigger is re-evaluated so it can RE-GATE (fall silent) if it stops firing.
       - a release whose committed cb0 != ground-truth cb0 during user speech = barge-in.
  3. Metrics, in two groups:
       PROXY (vs ground truth): rollback%, saved_ms, coverage
       BEHAVIOURAL:             barge_in%, mean release-lead, n_releases
     Rollback is reported but flagged as a proxy --- a different-but-plausible token is not
     necessarily unnatural.

Budget calibration: each trigger has one knob (threshold/tau). We bisect it on the eval set
so its rollback matches --budget (default 0.02), then report all triggers at that matched
rollback. Pass --no-calibrate to use fixed thresholds instead.

This harness computes the metrics offline (no live Moshi needed) so the whole comparison
table is producible from cached features. Audio generation is a separate opt-in step
(--emit-audio) that drives Moshi via the confirmed gate mechanism (probe_generation_trigger).

Usage:
    PYTHONPATH=src python scripts/run_demo.py \
        --pairs-dir pairs_eval/ --labels-dir handoff_labels_eval/ \
        --head head_v0.pt --device cuda --budget 0.02 \
        --triggers oracle,entropy,argmax,js,probe \
        --acceptance cb0,top5 --save-json demo_table.json
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_triggers as dt


# --------------------------- precompute frame states ---------------------------
def build_states(head, feats, frames, tokens, K, device, batch):
    """Return arrays needed by all triggers + ground truth, for one conversation.
    p0[N,C,V] cb0 dist (horizon 1), truth0[N,C], ent[N,C], hidden[N,D], activity[N,C],
    and frame indices fr_idx[N]."""
    import torch
    logV = float(np.log(2048.0))
    C = tokens.shape[0]; Tlen = tokens.shape[2]
    rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + K < Tlen]
    if not rows:
        return None
    fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])
    P0, TR, EN, HD = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(ft_row), batch):
            br = ft_row[s:s + batch]; bf = fr_idx[s:s + batch]
            x = torch.from_numpy(feats[br].astype(np.float32)).to(device)
            HD.append(x.cpu().numpy())
            p = torch.softmax(head(x)[:, 0, :, 0, :], dim=-1)     # [b,C,V] cb0 horizon1
            P0.append(p.cpu().numpy().astype(np.float32))
            EN.append((-(p * (p + 1e-12).log()).sum(-1)).cpu().numpy() / logV)
            TR.append(np.stack([tokens[:, 0, f + 1] for f in bf]))  # cb0 truth at t+1 [b,C]
    p0 = np.concatenate(P0); truth0 = np.concatenate(TR)
    ent = np.concatenate(EN); hidden = np.concatenate(HD)
    cb0 = tokens[:, 0, :]
    sil = [int(np.bincount(cb0[c]).argmax()) for c in range(C)]
    activity = np.stack([[int(cb0[c, f] != sil[c]) for c in range(C)] for f in fr_idx])
    return dict(p0=p0, truth0=truth0, ent=ent, hidden=hidden, activity=activity,
                fr_idx=fr_idx, sil=sil)


def frame_state(S, i):
    return dt.FrameState(
        t=int(S["fr_idx"][i]),
        p0_now=S["p0"][i],
        p0_prev=S["p0"][i - 1] if i > 0 else None,
        ent_now=S["ent"][i],
        hidden=S["hidden"][i],
        activity=S["activity"][i],
    )


# --------------------------- continuous loop + metrics ---------------------------
def run_trigger(trig, states_list, accept_fn=None, topk=5, ent_floor=0.5):
    """Continuous gate/release loop for one trigger.

    `accept_fn(p0,true)->[C]bool` defines a VALID release under the chosen acceptance rule
    (cb0 exact or top-k). It drives barge-in accounting so the cb0 and top5 rows genuinely
    DIFFER: under top5 a release whose true token is a confident top-5 candidate is not a
    barge-in; under cb0 it is. Reports exact-rollback and top5-miss both; their gap = plausible
    divergences. `coverage` replaces the meaningless per-frame lead for continuous gates:
    fraction of real handoffs with a release in [handoff-4, handoff]."""
    n_release = n_frames = n_bargein = n_rollback = n_top5_miss = n_invalid = 0
    ho_hit = ho_total = 0

    for S, ho in states_list:
        trig.reset()
        N = len(S["p0"])
        ho_set = sorted(int(h) for h in ho)
        released_frames = []
        for i in range(N):
            st = frame_state(S, i)
            fire = trig.fire(st)
            n_frames += 1
            if fire:
                n_release += 1
                released_frames.append(st.t)
                pred0 = S["p0"][i].argmax(-1)
                true0 = S["truth0"][i]
                exact_ok = (pred0 == true0)
                top5_ok = dt.accept_topk(S["p0"][i], true0, topk, ent_floor)
                valid = accept_fn(S["p0"][i], true0) if accept_fn is not None else exact_ok
                if not exact_ok.all():
                    n_rollback += 1
                if not top5_ok.all():
                    n_top5_miss += 1
                if not valid.all():
                    n_invalid += 1
                if S["activity"][i, 0] == 1 and not valid.all():
                    n_bargein += 1
        rel_set = set(released_frames)
        for h in ho_set:
            ho_total += 1
            if any((h - 4) <= r <= h for r in rel_set):
                ho_hit += 1

    tot = sum(len(S["p0"]) for S, _ in states_list)
    return dict(
        rollback=n_rollback / max(n_frames, 1),
        top5_miss=n_top5_miss / max(n_frames, 1),
        invalid=n_invalid / max(n_release, 1),
        barge_in=n_bargein / max(n_release, 1),
        saved_ms=80.0 * n_release / max(tot, 1),
        n_release=n_release,
        coverage=ho_hit / max(ho_total, 1),
        release_rate=n_release / max(n_frames, 1),
    )


def calibrate(trig_ctor, states_list, budget, knob_lo, knob_hi, accept_fn=None,
              iters=12, higher_knob_more_release=True, target="top5_miss"):
    """Bisect the trigger's knob so `target` ~= budget. Calibrates on top5_miss by default
    (the honest 'genuinely wrong' signal), NOT exact rollback --- a different-but-plausible
    token should not count against the budget. Returns (knob, metrics)."""
    lo, hi = knob_lo, knob_hi
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = run_trigger(trig_ctor(mid), states_list, accept_fn=accept_fn)
        best = (mid, m)
        if m[target] > budget:                    # too many genuine misses -> tighten
            if higher_knob_more_release:
                hi = mid
            else:
                lo = mid
        else:
            if higher_knob_more_release:
                lo = mid
            else:
                hi = mid
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--budget", type=float, default=0.02, help="target rollback budget")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--triggers", default="oracle,entropy,argmax,js,probe")
    ap.add_argument("--acceptance", default="cb0,top5")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ent-floor", type=float, default=0.5)
    ap.add_argument("--probe", type=Path, help="probe .npz for the probe trigger")
    ap.add_argument("--save-json", type=Path)
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from duplex_spec.head import MultiStepTPPHead, MultiStepDepHead
    ck = torch.load(args.head, map_location=args.device)
    Head = MultiStepDepHead if ck.get("head_type") == "dep" else MultiStepTPPHead
    head = Head(hidden_dim=ck["hidden_dim"], n_channels=ck["n_channels"],
                n_codebooks=ck["n_codebooks"], codebook_size=2048, horizon=ck["horizon"])
    head.load_state_dict(ck["state_dict"]); head.to(args.device).eval()
    K = ck["horizon"]

    # load conversations + handoff labels
    states_list = []
    for npz in sorted(args.pairs_dir.glob("*.npz")):
        npy = npz.with_suffix(".npy")
        lab = args.labels_dir / f"{npz.stem}.npz"
        if not (npy.exists() and lab.exists()):
            continue
        d = np.load(npz); tk = np.load(npy)
        S = build_states(head, d["feats"], d["frames"], tk, K, args.device, args.batch)
        if S is None:
            continue
        ho = np.load(lab)["handoff_frames"]
        states_list.append((S, ho))
    if not states_list:
        sys.exit("no conversations with matching labels found")
    print(f"[data] {len(states_list)} conversations\n")

    triggers = args.triggers.split(",")
    accs = args.acceptance.split(",")

    def accept_for(rule):
        if rule == "cb0":
            return lambda p0, tru: dt.accept_cb0(p0, tru)
        k = int(rule[3:])
        return lambda p0, tru: dt.accept_topk(p0, tru, k, args.ent_floor)

    # gather all handoff frames for oracle
    all_ho = [ho for _, ho in states_list]

    rows = []
    for tname in triggers:
        variants = accs if tname in ("argmax", "js", "probe") else [None]
        for rule in variants:
            acc_fn = accept_for(rule) if rule else None
            # build trigger (+ calibrate knob for gate/entropy triggers)
            if tname == "oracle":
                trig = dt.OracleTrigger(handoff_frames=np.concatenate(all_ho) if all_ho else [],
                                        lead=2)
                # oracle uses per-conversation labels; rebuild per conv in loop instead:
                m = run_oracle(states_list, lead=2, accept_fn=acc_fn)
                label = "oracle"
            elif tname == "entropy":
                ctor = lambda thr: dt.EntropyTrigger(threshold=thr)
                knob, m = (0.5, run_trigger(ctor(0.5), states_list)) if args.no_calibrate \
                    else calibrate(ctor, states_list, args.budget, 0.05, 0.9,
                                   higher_knob_more_release=True)
                label = "entropy"
            elif tname == "argmax":
                ctor = lambda thr: dt.ArgmaxGate(ent_floor=thr)
                knob, m = (0.5, run_trigger(ctor(0.5), states_list, acc_fn)) if args.no_calibrate \
                    else calibrate(lambda t: dt.ArgmaxGate(ent_floor=t), states_list,
                                   args.budget, 0.1, 0.9, acc_fn, higher_knob_more_release=True)
                m = run_trigger(dt.ArgmaxGate(ent_floor=knob), states_list, acc_fn)
                label = f"argmax·{rule}"
            elif tname == "js":
                ctor = lambda thr: dt.JSGate(threshold=thr, ent_floor=args.ent_floor)
                knob, m = (0.1, run_trigger(ctor(0.1), states_list, acc_fn)) if args.no_calibrate \
                    else calibrate(ctor, states_list, args.budget, 0.001, 0.6, acc_fn,
                                   higher_knob_more_release=True)
                m = run_trigger(ctor(knob), states_list, acc_fn)
                label = f"js·{rule}"
            elif tname == "probe":
                if not args.probe:
                    print("[skip] probe trigger needs --probe"); continue
                pr = np.load(args.probe, allow_pickle=True)
                score_fn = make_probe_score(pr)
                ctor = lambda thr: dt.ProbeGate(score_fn=score_fn, threshold=thr)
                knob, m = (0.5, run_trigger(ctor(0.5), states_list, acc_fn)) if args.no_calibrate \
                    else calibrate(ctor, states_list, args.budget, 0.1, 0.95, acc_fn,
                                   higher_knob_more_release=False)
                m = run_trigger(ctor(knob), states_list, acc_fn)
                label = f"probe·{rule}"
            else:
                print(f"[skip] unknown trigger {tname}"); continue
            rows.append((label, m))

    # print table: BOTH rollback (strict ruler) and top5_miss (honest signal) side by side
    print(f"{'trigger':>14} {'saved_ms':>9} {'exact_rb':>9} {'top5_miss':>10} "
          f"{'barge_in':>9} {'coverage':>9} {'releases':>9}")
    print("-" * 74)
    out = {}
    for label, m in rows:
        print(f"{label:>14} {m['saved_ms']:>9.1f} {m['rollback']:>8.1%} "
              f"{m['top5_miss']:>9.1%} {m['barge_in']:>8.1%} "
              f"{m['coverage']:>8.1%} {m['n_release']:>9}")
        out[label] = m
    print("\n  exact_rb  = committed cb0 != ground-truth cb0 (STRICT RULER; kept for")
    print("              comparability, but a different-but-plausible token is not a failure).")
    print("  top5_miss = true token OUTSIDE the confident top-5 (the honest 'genuinely")
    print("              wrong' signal). The exact_rb - top5_miss gap = plausible divergences.")
    print("  barge_in  = released while user active AND INVALID under the chosen rule")
    print("              (so cb0 and top5 rows now differ: top5 forgives plausible tokens).")
    print("  coverage  = fraction of real handoffs with a release in [handoff-4, handoff].")
    print(f"  Calibrated to top5_miss budget = {args.budget:.0%}." if not args.no_calibrate
          else "  (fixed thresholds)")
    if args.save_json:
        args.save_json.write_text(json.dumps(out, indent=2))
        print(f"\n[json] {args.save_json}")


def run_oracle(states_list, lead, accept_fn, topk=5, ent_floor=0.5):
    """Oracle fires `lead` frames before each ground-truth handoff. Because it IS handoff-
    aligned, mean_lead_ms is meaningful here (unlike continuous gates). Reports the same
    columns plus coverage."""
    n_release = n_frames = n_bargein = n_rollback = n_top5_miss = n_invalid = 0
    ho_hit = ho_total = 0
    for S, ho in states_list:
        ho_set = set(int(h) - lead for h in ho)
        rel = []
        N = len(S["p0"])
        for i in range(N):
            n_frames += 1
            if int(S["fr_idx"][i]) in ho_set:
                n_release += 1; rel.append(int(S["fr_idx"][i]))
                pred0 = S["p0"][i].argmax(-1); true0 = S["truth0"][i]
                exact_ok = (pred0 == true0)
                top5_ok = dt.accept_topk(S["p0"][i], true0, topk, ent_floor)
                valid = accept_fn(S["p0"][i], true0) if accept_fn else exact_ok
                if not exact_ok.all(): n_rollback += 1
                if not top5_ok.all(): n_top5_miss += 1
                if not valid.all(): n_invalid += 1
                if S["activity"][i, 0] == 1 and not valid.all(): n_bargein += 1
        relset = set(rel)
        for h in (int(x) for x in ho):
            ho_total += 1
            if any((h - 4) <= r <= h for r in relset): ho_hit += 1
    tot = sum(len(S["p0"]) for S, _ in states_list)
    return dict(rollback=n_rollback / max(n_frames, 1),
                top5_miss=n_top5_miss / max(n_frames, 1),
                invalid=n_invalid / max(n_release, 1),
                barge_in=n_bargein / max(n_release, 1),
                saved_ms=80.0 * n_release / max(tot, 1), n_release=n_release,
                coverage=ho_hit / max(ho_total, 1),
                release_rate=n_release / max(n_frames, 1))

def make_probe_score(pr):
    w, b, mu, sd = pr["w"], float(pr["b"]), pr["mu"], pr["sd"]
    FEAT = list(pr["feat_names"]) if "feat_names" in pr else None

    def score(st):
        # reconstruct the probe's features from FrameState (cb0 dist etc.)
        p0 = st.p0_now
        srt = np.sort(p0, -1)
        ent = st.ent_now.mean()
        top1 = srt[:, -1].mean(); top2 = srt[:, -2].mean()
        margin = top1 - top2
        js_prev = 0.0 if st.p0_prev is None else float(dt.js_div(p0, st.p0_prev).mean())
        has_hist = 0.0 if st.p0_prev is None else 1.0
        # feature order must match train_probe.py FEAT_NAMES
        feat = np.array([ent, top1, margin, 0.0, js_prev, 0.0, 0.0, has_hist])
        z = ((feat - mu) / sd) @ w + b
        return float(1 / (1 + np.exp(-z)))
    return score


if __name__ == "__main__":
    main()
