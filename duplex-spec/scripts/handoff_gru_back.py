"""Strong handoff predictor (v2) --- GRU over a window of rich features, regressing
time-to-next-handoff. Replaces the weak logistic-regression-on-binary-activity model
(AUC 0.62) that could not fire early with usable precision.

THREE upgrades over the weak version:
  1. FEATURES. Not just binary activity. Per frame we use:
       - backbone hidden state h_t (4096-d), projected to a small dim by a learned linear
         (the untapped signal: prosody/uncertainty Moshi already encoded)
       - head cb0 distribution stats: entropy, top1 prob, top1-top2 margin, per channel
       - activity: active flags, per-channel silence fraction, turn length
  2. TARGET. Regress time-to-next-handoff (in frames, capped), not the rare binary
     "handoff in <=H". Dense target -> no 1-4% class-imbalance collapse. At inference we
     fire when predicted time-to-handoff <= a lead threshold.
  3. MODEL. A GRU over the last N frames (turn-endings are a trajectory, not a snapshot),
     which the LR could not represent.

Labels come from candor_handoffs.py (real backbiter turn boundaries), same as the weak
version, so the comparison is clean.

Usage:
    # fit
    PYTHONPATH=src python scripts/handoff_gru.py fit \
        --pairs-dir pairs/ --labels-dir handoff_labels/ \
        --win 24 --cap 25 --out handoff_gru.pt --device cuda
    # eval early-warning curve
    PYTHONPATH=src python scripts/handoff_gru.py eval \
        --pairs-dir pairs_eval/ --labels-dir handoff_labels_eval/ \
        --model handoff_gru.pt --device cuda
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np


# --------------------------- feature extraction ---------------------------
def conv_features(head, feats, frames, tokens, K, device, batch, hproj_dim=None):
    """Per-frame feature matrix [T_kept, F] and the kept frame indices.
    Features: [entropy_c0, entropy_c1, top1_c0, top1_c1, margin_c0, margin_c1,
               act_c0, act_c1, silfrac_c0, silfrac_c1, turnlen] ++ (optional h_t raw)
    The h_t raw block is returned separately so the model can project it.
    """
    import torch
    logV = float(np.log(2048.0))
    C = tokens.shape[0]
    Tlen = tokens.shape[2]
    rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + K < Tlen]
    if not rows:
        return None
    fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])

    dist_feats = []; hstates = []
    with torch.no_grad():
        for s in range(0, len(ft_row), batch):
            br = ft_row[s:s + batch]
            x = torch.from_numpy(feats[br].astype(np.float32)).to(device)
            hstates.append(x.cpu().numpy())
            lo = head(x)                                   # [b,K,C,Q,V]
            p0 = torch.softmax(lo[:, 0, :, 0, :], dim=-1)  # cb0 of horizon 1: [b,C,V]
            ent = (-(p0 * (p0 + 1e-12).log()).sum(-1)).cpu().numpy() / logV   # [b,C]
            srt = torch.sort(p0, dim=-1).values
            top1 = srt[..., -1].cpu().numpy(); top2 = srt[..., -2].cpu().numpy()
            dist_feats.append(np.concatenate([ent, top1, top1 - top2], axis=1))  # [b, 3C]
    dist = np.concatenate(dist_feats)                      # [T_kept, 3C]
    hmat = np.concatenate(hstates)                         # [T_kept, 4096]

    # activity features aligned to kept frames
    cb0 = tokens[:, 0, :]
    sil = [int(np.bincount(cb0[c]).argmax()) for c in range(C)]
    act = np.stack([cb0[c] != sil[c] for c in range(C)]).astype(np.float32)   # [C, Tlen]
    # dominant + turn length
    dom = np.full(Tlen, -1); cur = -1
    for t in range(Tlen):
        a0, a1 = act[0, t], act[1, t]
        if a0 and not a1: cur = 0
        elif a1 and not a0: cur = 1
        dom[t] = cur
    turnlen = np.zeros(Tlen); tl = 0
    for t in range(Tlen):
        if t > 0 and dom[t] != dom[t - 1] and dom[t] != -1: tl = 0
        tl += 1; turnlen[t] = tl

    act_feats = []
    N = 12
    for f in fr_idx:
        lo_ = max(0, f - N + 1); w = act[:, lo_:f + 1]
        act_feats.append([act[0, f], act[1, f],
                          1 - w[0].mean(), 1 - w[1].mean(),
                          min(turnlen[f] / 50.0, 2.0)])
    act_feats = np.asarray(act_feats, np.float32)          # [T_kept, 5]

    feat = np.concatenate([dist, act_feats], axis=1)       # [T_kept, 3C+5]
    return feat.astype(np.float32), hmat.astype(np.float32), fr_idx


def time_to_handoff(handoff_frames, fr_idx, cap):
    """Regression target: frames until the next handoff at/after each kept frame, capped."""
    ho = np.sort(np.asarray(handoff_frames))
    tt = np.full(len(fr_idx), cap, np.float32)
    for i, f in enumerate(fr_idx):
        nxt = ho[ho >= f]
        if len(nxt):
            tt[i] = min(cap, int(nxt[0] - f))
    return tt


# --------------------------------- model ---------------------------------
def build_model(torch, n_feat, hproj_dim, gru_hidden):
    import torch.nn as nn

    class HandoffGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.hproj = nn.Linear(4096, hproj_dim)
            self.norm = nn.LayerNorm(n_feat + hproj_dim)
            self.gru = nn.GRU(n_feat + hproj_dim, gru_hidden, batch_first=True)
            self.out = nn.Linear(gru_hidden, 1)            # predicts time-to-handoff

        def forward(self, feat_seq, h_seq):
            # feat_seq [B,W,n_feat], h_seq [B,W,4096]
            h = self.hproj(h_seq)
            x = torch.cat([feat_seq, h], dim=-1)
            x = self.norm(x)
            o, _ = self.gru(x)
            return self.out(o[:, -1, :]).squeeze(-1)        # [B] predicted time-to-handoff
    return HandoffGRU()


def make_windows(feat, hmat, target, win):
    """Sliding windows of length `win`; label is the target at the window's LAST frame."""
    T = len(feat)
    if T < win:
        return None
    Fw = np.stack([feat[i - win + 1:i + 1] for i in range(win - 1, T)])       # [M,win,nf]
    Hw = np.stack([hmat[i - win + 1:i + 1] for i in range(win - 1, T)])       # [M,win,4096]
    Yw = target[win - 1:]                                                     # [M]
    return Fw, Hw, Yw


# --------------------------------- commands ---------------------------------
def load_head(head_path, device):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from duplex_spec.head import MultiStepTPPHead, MultiStepDepHead
    ck = torch.load(head_path, map_location=device)
    Head = MultiStepDepHead if ck.get("head_type") == "dep" else MultiStepTPPHead
    head = Head(hidden_dim=ck["hidden_dim"], n_channels=ck["n_channels"],
                n_codebooks=ck["n_codebooks"], codebook_size=2048, horizon=ck["horizon"])
    head.load_state_dict(ck["state_dict"]); head.to(device).eval()
    return head, ck["horizon"]


def iter_convs(pairs_dir, labels_dir):
    for npz in sorted(Path(pairs_dir).glob("*.npz")):
        npy = npz.with_suffix(".npy")
        lab = Path(labels_dir) / f"{npz.stem}.npz"
        if npy.exists() and lab.exists():
            yield npz.stem, npz, npy, lab


def cmd_fit(args):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    head, K = load_head(args.head, args.device)
    Fw_all, Hw_all, Yw_all = [], [], []
    for cid, npz, npy, lab in iter_convs(args.pairs_dir, args.labels_dir):
        d = np.load(npz); tk = np.load(npy); L = np.load(lab)
        r = conv_features(head, d["feats"], d["frames"], tk, K, args.device, args.batch)
        if r is None: continue
        feat, hmat, fr_idx = r
        tgt = time_to_handoff(L["handoff_frames"], fr_idx, args.cap)
        w = make_windows(feat, hmat, tgt, args.win)
        if w is None: continue
        Fw_all.append(w[0]); Hw_all.append(w[1]); Yw_all.append(w[2])
        print(f"[data] {cid}: {len(w[2])} windows")
    Fw = np.concatenate(Fw_all); Hw = np.concatenate(Hw_all); Yw = np.concatenate(Yw_all)
    print(f"[fit] {len(Fw)} windows, target mean={Yw.mean():.1f} (cap={args.cap})")

    n_feat = Fw.shape[-1]
    model = build_model(torch, n_feat, args.hproj_dim, args.gru_hidden).to(args.device)
    # standardise features (not h_t; LayerNorm handles the concat)
    mu = Fw.reshape(-1, n_feat).mean(0); sd = Fw.reshape(-1, n_feat).std(0) + 1e-6
    Fw = (Fw - mu) / sd
    ds = TensorDataset(torch.tensor(Fw), torch.tensor(Hw), torch.tensor(Yw))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.SmoothL1Loss()
    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for fb, hb, yb in dl:
            fb, hb, yb = fb.to(args.device), hb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            pred = model(fb, hb)
            loss = lossf(pred, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(fb); n += len(fb)
        print(f"[ep {ep}] train_smoothL1={tot/n:.3f}")
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "win": args.win,
                "cap": args.cap, "n_feat": n_feat, "hproj_dim": args.hproj_dim,
                "gru_hidden": args.gru_hidden, "head": str(args.head)}, args.out)
    print(f"[save] {args.out}")


def cmd_eval(args):
    import torch
    m = torch.load(args.model, map_location=args.device)
    head, K = load_head(Path(m["head"]) if Path(m["head"]).exists() else args.head, args.device)
    model = build_model(torch, m["n_feat"], m["hproj_dim"], m["gru_hidden"]).to(args.device)
    model.load_state_dict(m["state_dict"]); model.eval()
    mu, sd, win, cap = m["mu"], m["sd"], m["win"], m["cap"]

    preds, trues = [], []
    for cid, npz, npy, lab in iter_convs(args.pairs_dir, args.labels_dir):
        d = np.load(npz); tk = np.load(npy); L = np.load(lab)
        r = conv_features(head, d["feats"], d["frames"], tk, K, args.device, args.batch)
        if r is None: continue
        feat, hmat, fr_idx = r
        tgt = time_to_handoff(L["handoff_frames"], fr_idx, cap)
        w = make_windows(feat, hmat, tgt, win)
        if w is None: continue
        Fw = (w[0] - mu) / sd
        with torch.no_grad():
            p = model(torch.tensor(Fw).to(args.device),
                      torch.tensor(w[1]).to(args.device)).cpu().numpy()
        preds.append(p); trues.append(w[2])
    P = np.concatenate(preds); Y = np.concatenate(trues)

    # early-warning: "handoff imminent" = true time-to-handoff <= lead. We FIRE when
    # predicted time <= lead. Sweep lead; report precision/recall + mean actual lead.
    print(f"[eval] {len(P)} windows  |  MAE={np.abs(P-Y).mean():.2f} frames "
          f"({80*np.abs(P-Y).mean():.0f} ms)\n")
    print(f"{'lead(fr)':>8} {'lead(ms)':>8} {'precision':>10} {'recall':>8} {'fire_rate':>10}")
    print("-" * 48)
    out = {"mae_frames": float(np.abs(P - Y).mean())}
    for lead in [int(x) for x in args.leads.split(",")]:
        imminent = Y <= lead
        fire = P <= lead
        tp = int((fire & imminent).sum()); fp = int((fire & ~imminent).sum())
        fn = int((~fire & imminent).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"{lead:>8} {80*lead:>8} {prec:>9.1%} {rec:>7.1%} {fire.mean():>9.1%}")
        out[f"lead{lead}"] = dict(precision=prec, recall=rec, fire_rate=float(fire.mean()))
    print("\n  precision = of fires, how many were real imminent handoffs (1-prec = barge-in).")
    print("  Compare to the weak LR (AUC 0.62, ~5% precision). MAE in ms is the headline:")
    print("  how far off, on average, the predicted time-to-handoff is.")
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"[json] {args.save_json}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--pairs-dir", type=Path, required=True)
        s.add_argument("--labels-dir", type=Path, required=True)
        s.add_argument("--head", type=Path, default=Path("head_v0.pt"))
        s.add_argument("--device", default="cuda")
        s.add_argument("--batch", type=int, default=256)
        s.add_argument("--win", type=int, default=24, help="GRU window length (frames)")
        s.add_argument("--cap", type=int, default=25, help="max time-to-handoff (frames)")
        s.add_argument("--hproj-dim", type=int, default=64, help="h_t projection dim")
        s.add_argument("--gru-hidden", type=int, default=128)
        if name == "fit":
            s.add_argument("--epochs", type=int, default=15)
            s.add_argument("--lr", type=float, default=1e-3)
            s.add_argument("--out", type=Path, default=Path("handoff_gru.pt"))
        else:
            s.add_argument("--model", type=Path, required=True)
            s.add_argument("--leads", default="1,2,3,4,6,8", help="lead times (frames) to sweep")
            s.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()
    (cmd_fit if args.cmd == "fit" else cmd_eval)(args)


if __name__ == "__main__":
    main()
