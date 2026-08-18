"""Acoustic handoff predictor --- reads the RAW codec tokens (all 8 codebooks, both
channels) over a window and embeds them, instead of the head's cb0 prediction stats.

WHY
---
The head-features version (handoff_gru.py) reached ~20-31% precision but low recall: the
head's cb0 prediction is a lossy shadow of the acoustics. Turn-end cues (falling pitch,
slowing tempo, final lengthening, energy taper) live in the codec tokens themselves --- that
is what Mimi encodes. This model reads those tokens directly, the way VAP-style timing models
read the acoustic stream, so it sees the prosody the head-stats discard.

Crucially this needs NO head and NO cached features --- only the token .npy files. Each frame
is 16 tokens (8 codebooks x 2 channels); we learn a per-(channel,codebook) embedding, sum
them into a frame vector, and run a GRU over the window. Target is time-to-next-handoff
(same as handoff_gru.py) so the two are directly comparable --- the ONLY change is the input
signal, isolating whether acoustic features carry the handoff signal the head-stats missed.

Usage:
    PYTHONPATH=src python scripts/handoff_gru_acoustic.py fit \
        --tokens-dir tokens/ --labels-dir handoff_labels/ \
        --win 24 --cap 25 --emb-dim 64 --out handoff_acoustic.pt --device cuda
    PYTHONPATH=src python scripts/handoff_gru_acoustic.py eval \
        --tokens-dir tokens_eval/ --labels-dir handoff_labels_eval/ \
        --model handoff_acoustic.pt --device cuda
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np


def time_to_handoff(handoff_frames, T, cap):
    """Dense regression target over ALL frames 0..T-1: frames to next handoff, capped."""
    ho = np.sort(np.asarray(handoff_frames))
    tt = np.full(T, cap, np.float32)
    j = 0
    for t in range(T):
        while j < len(ho) and ho[j] < t:
            j += 1
        if j < len(ho):
            tt[t] = min(cap, ho[j] - t)
    return tt


def iter_token_convs(tokens_dir, labels_dir):
    for npy in sorted(Path(tokens_dir).glob("*.npy")):
        lab = Path(labels_dir) / f"{npy.stem}.npz"
        if lab.exists():
            yield npy.stem, npy, lab


def build_model(torch, C, Q, V, emb_dim, gru_hidden, aux_dim=2):
    import torch.nn as nn

    class AcousticHandoffGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(V + 1, emb_dim) for _ in range(C * Q)])
            self.C, self.Q = C, Q
            self.norm = nn.LayerNorm(emb_dim + aux_dim)
            self.gru = nn.GRU(emb_dim + aux_dim, gru_hidden, batch_first=True)
            self.out = nn.Linear(gru_hidden, 1)

        def forward(self, tok_seq, aux_seq):
            B, W, C, Q = tok_seq.shape
            flat = tok_seq.reshape(B, W, C * Q)
            acc = 0
            for i in range(C * Q):
                acc = acc + self.emb[i](flat[:, :, i])
            frame = acc / (C * Q)
            x = torch.cat([frame, aux_seq], dim=-1)
            x = self.norm(x)
            o, _ = self.gru(x)
            return self.out(o[:, -1, :]).squeeze(-1)
    return AcousticHandoffGRU()


def activity_stream(tokens):
    C = tokens.shape[0]
    cb0 = tokens[:, 0, :]
    sil = [int(np.bincount(cb0[c]).argmax()) for c in range(C)]
    return np.stack([(cb0[c] != sil[c]).astype(np.float32) for c in range(C)])


def cmd_fit(args):
    import torch
    from torch.utils.data import DataLoader
    tok_list, aux_list, tgt_list, index = [], [], [], []
    C = Q = None
    for cid, npy, lab in iter_token_convs(args.tokens_dir, args.labels_dir):
        tokens = np.load(npy)
        C, Q, T = tokens.shape
        Quse = min(Q, args.max_cb)
        ho = np.load(lab)["handoff_frames"]
        tgt = time_to_handoff(ho, T, args.cap)
        act = activity_stream(tokens)
        if T < args.win:
            continue
        ci = len(tok_list)
        tok_list.append(tokens[:, :Quse, :].astype(np.int64))
        aux_list.append(act.astype(np.float32))
        tgt_list.append(tgt)
        index += [(ci, e) for e in range(args.win - 1, T)]
        print(f"[data] {cid}: {T - args.win + 1} windows")
    if not index:
        sys.exit("no windows")
    Quse = min(Q, args.max_cb)
    print(f"[fit] {len(index)} windows (lazy), C={C} Q_used={Quse} cap={args.cap}")
    win = args.win

    class WinDS(torch.utils.data.Dataset):
        def __len__(self): return len(index)
        def __getitem__(self, k):
            ci, e = index[k]
            tk = tok_list[ci][:, :, e - win + 1:e + 1]
            tk = np.transpose(tk, (2, 0, 1))
            aux = aux_list[ci][:, e - win + 1:e + 1].T
            y = tgt_list[ci][e]
            return (torch.from_numpy(tk.copy()).long(),
                    torch.from_numpy(aux.copy()).float(),
                    torch.tensor(float(y)))

    model = build_model(torch, C, Quse, 2048, args.emb_dim, args.gru_hidden, aux_dim=C).to(args.device)
    dl = DataLoader(WinDS(), batch_size=args.batch, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.SmoothL1Loss()
    print(f"[model] acoustic GRU, emb={args.emb_dim} params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for tk, aux, yb in dl:
            tk, aux, yb = tk.to(args.device), aux.to(args.device), yb.to(args.device)
            opt.zero_grad()
            loss = lossf(model(tk, aux), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(tk); n += len(tk)
        print(f"[ep {ep}] train_smoothL1={tot/n:.3f}")
    torch.save({"state_dict": model.state_dict(), "win": win, "cap": args.cap,
                "C": C, "Q": Quse, "emb_dim": args.emb_dim, "gru_hidden": args.gru_hidden},
               args.out)
    print(f"[save] {args.out}")


def cmd_eval(args):
    import torch
    m = torch.load(args.model, map_location=args.device, weights_only=False)
    C, Q, win, cap = m["C"], m["Q"], m["win"], m["cap"]
    model = build_model(torch, C, Q, 2048, m["emb_dim"], m["gru_hidden"], aux_dim=C).to(args.device)
    model.load_state_dict(m["state_dict"]); model.eval()
    preds, trues = [], []
    for cid, npy, lab in iter_token_convs(args.tokens_dir, args.labels_dir):
        tokens = np.load(npy); T = tokens.shape[2]
        if T < win: continue
        ho = np.load(lab)["handoff_frames"]
        tgt = time_to_handoff(ho, T, cap)
        act = activity_stream(tokens)
        tk_all = tokens[:, :Q, :].astype(np.int64)
        ends = list(range(win - 1, T))
        for s0 in range(0, len(ends), args.batch):
            be = ends[s0:s0 + args.batch]
            tkb = np.stack([np.transpose(tk_all[:, :, e - win + 1:e + 1], (2, 0, 1)) for e in be])
            auxb = np.stack([act[:, e - win + 1:e + 1].T for e in be])
            with torch.no_grad():
                p = model(torch.from_numpy(tkb).long().to(args.device),
                          torch.from_numpy(auxb).float().to(args.device)).cpu().numpy()
            preds.append(p); trues.append(np.array([tgt[e] for e in be], np.float32))
    P = np.concatenate(preds); Y = np.concatenate(trues)
    print(f"[eval] {len(P)} windows  |  MAE={np.abs(P-Y).mean():.2f} frames "
          f"({80*np.abs(P-Y).mean():.0f} ms)\n")
    print(f"{'lead(fr)':>8} {'lead(ms)':>8} {'precision':>10} {'recall':>8} {'fire_rate':>10}")
    print("-" * 48)
    out = {"mae_frames": float(np.abs(P - Y).mean())}
    for lead in [int(x) for x in args.leads.split(",")]:
        imminent = Y <= lead; fire = P <= lead
        tp = int((fire & imminent).sum()); fp = int((fire & ~imminent).sum())
        fn = int((~fire & imminent).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"{lead:>8} {80*lead:>8} {prec:>9.1%} {rec:>7.1%} {fire.mean():>9.1%}")
        out[f"lead{lead}"] = dict(precision=prec, recall=rec, fire_rate=float(fire.mean()))
    print("\n  Compare to head-features GRU (prec ~20-31%, recall ~1-4%). If acoustic tokens")
    print("  lift precision AND recall, the prosody signal the head-stats missed is real.")
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"[json] {args.save_json}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--tokens-dir", type=Path, required=True)
        s.add_argument("--labels-dir", type=Path, required=True)
        s.add_argument("--device", default="cuda")
        s.add_argument("--batch", type=int, default=256)
        s.add_argument("--win", type=int, default=24)
        s.add_argument("--cap", type=int, default=25)
        s.add_argument("--max-cb", type=int, default=8)
        s.add_argument("--emb-dim", type=int, default=64)
        s.add_argument("--gru-hidden", type=int, default=128)
        if name == "fit":
            s.add_argument("--epochs", type=int, default=15)
            s.add_argument("--lr", type=float, default=1e-3)
            s.add_argument("--out", type=Path, default=Path("handoff_acoustic.pt"))
        else:
            s.add_argument("--model", type=Path, required=True)
            s.add_argument("--leads", default="1,2,3,4,6,8")
            s.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()
    (cmd_fit if args.cmd == "fit" else cmd_eval)(args)


if __name__ == "__main__":
    main()
