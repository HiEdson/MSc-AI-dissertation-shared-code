"""Train the multi-step head on cached Moshi features + Stage-A tokens.

Inputs per conversation:
  feats .npz  (Stage B): feats [M, H], frames [M]
  tokens .npy (Stage A): [n_channels, n_codebooks, T]
A training example is (feature[j]) -> (token-pairs at frames frames[j]+1 .. +K).

Diagnostics each epoch:
  - per-horizon-step accuracy (k1..kK): genuine anticipation should score higher
    for near frames than far ones.
  - COPY baseline: predict the CURRENT frame's tokens for every future frame.
    The head must beat this; copy is high during silence/pauses, so beating it
    is what shows the head learns real dynamics, not "repeat what's here".
Saves the BEST checkpoint by val loss (not the last, which overfits).

The frozen backbone is NOT loaded here — trains in minutes on cached features.

Usage:
    python train_head.py --feats feats/<c>.npz --tokens tokens/<c>.npy --horizon 4 --epochs 10
    python train_head.py --pairs-dir <dir with matching .npz + .npy> --horizon 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_pair(feats_path: Path, tokens_path: Path, horizon: int, target_shift: int = 0):
    """Return (X [N,H], Y [N,C,Q,K], Y0 [N,C,Q]) for one conversation.

    Y  = future token-pairs starting at frame f+1+target_shift (targets).
    Y0 = the current observed frame's tokens (copy baseline; unaffected by shift).
    target_shift lets you correct a feature<->target frame offset (e.g. acoustic
    delay): shift=+1 means feature[f] predicts frames f+2..f+1+K instead.
    """
    d = np.load(feats_path)
    feats, frames = d["feats"], d["frames"]
    tokens = np.load(tokens_path)                       # [C, Q, T]
    C, Q, T = tokens.shape
    rows, fs = [], []
    for row, f in enumerate(frames):
        f = int(f); start = f + 1 + target_shift
        if start >= 0 and start + horizon <= T:
            rows.append(row); fs.append(f)
    if not rows:
        z = np.zeros((0, feats.shape[1]), np.float32)
        return z, np.zeros((0, C, Q, horizon), np.int64), np.zeros((0, C, Q), np.int64)
    X = np.stack([feats[r] for r in rows]).astype(np.float32)
    Y = np.stack([tokens[:, :, f + 1 + target_shift: f + 1 + target_shift + horizon] for f in fs]).astype(np.int64)
    Y0 = np.stack([tokens[:, :, f] for f in fs]).astype(np.int64)
    return X, Y, Y0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=Path, action="append", default=[])
    ap.add_argument("--tokens", type=Path, action="append", default=[])
    ap.add_argument("--pairs-dir", type=Path, help="dir with <stem>.npz feats + <stem>.npy tokens")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--target-shift", type=int, default=0, help="frame offset feature->target (try -2..+2)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("head.pt"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    from duplex_spec.head import MultiStepTPPHead, tpp_loss

    pairs = list(zip(args.feats, args.tokens))
    if args.pairs_dir:
        for npz in sorted(args.pairs_dir.glob("*.npz")):
            npy = npz.with_suffix(".npy")
            if npy.exists():
                pairs.append((npz, npy))
    if not pairs:
        sys.exit("No (feats, tokens) pairs. Use --feats/--tokens or --pairs-dir.")

    Xs, Ys, Y0s = [], [], []
    for fp, tp in pairs:
        X, Y, Y0 = load_pair(fp, tp, args.horizon, args.target_shift)
        if len(X):
            Xs.append(X); Ys.append(Y); Y0s.append(Y0)
        print(f"[data] {fp.name}: {len(X)} examples")
    X = np.concatenate(Xs); Y = np.concatenate(Ys); Y0 = np.concatenate(Y0s)
    H = X.shape[1]; C, Q, K = Y.shape[1], Y.shape[2], Y.shape[3]
    print(f"[data] total {len(X)} examples  H={H} C={C} Q={Q} K={K} target_shift={args.target_shift}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * args.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    dev = args.device
    Xt = torch.from_numpy(X).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    Y0t = torch.from_numpy(Y0).to(dev)
    head = MultiStepTPPHead(hidden_dim=H, n_channels=C, n_codebooks=Q,
                            codebook_size=2048, horizon=K).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    print(f"[head] params={sum(p.numel() for p in head.parameters())/1e6:.1f}M")

    def per_step_acc(pred_kcq, tgt_kcq):
        # both [N, K, C, Q]; return list of K accuracies (avg over N,C,Q)
        eq = (pred_kcq == tgt_kcq).float()
        return [eq[:, k].mean().item() for k in range(K)]

    def per_cb_acc(pred_kcq, tgt_kcq):
        # per codebook q, averaged over N, K (horizon) and C (channels)
        eq = (pred_kcq == tgt_kcq).float()           # [N,K,C,Q]
        return eq.mean(dim=(0, 1, 2)).tolist()        # [Q]

    def evaluate(rows):
        head.eval()
        with torch.no_grad():
            lo = head(Xt[rows])                          # [N,K,C,Q,V]
            loss = tpp_loss(lo, Yt[rows]).item()
            tgt = Yt[rows].permute(0, 3, 1, 2)           # [N,K,C,Q]
            pred = lo.argmax(-1)
            copy_pred = Y0t[rows][:, None].expand(-1, K, -1, -1)  # [N,K,C,Q]
            head_acc = per_step_acc(pred, tgt)
            copy_acc = per_step_acc(copy_pred, tgt)
            head_cb = per_cb_acc(pred, tgt)
            copy_cb = per_cb_acc(copy_pred, tgt)
        head.train()
        return loss, head_acc, copy_acc, head_cb, copy_cb

    def fmt(xs):
        return " ".join(f"{x:.3f}" for x in xs)

    tr = torch.as_tensor(tr_idx, device=dev)
    val = torch.as_tensor(val_idx, device=dev)
    best_val = float("inf")
    best_cb = None
    for ep in range(args.epochs):
        perm = tr[torch.randperm(len(tr), device=dev)]
        tot = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            opt.zero_grad()
            loss = tpp_loss(head(Xt[b]), Yt[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        vl, hacc, cacc, hcb, ccb = evaluate(val)
        flag = ""
        if vl < best_val:
            best_val = vl
            best_cb = (hcb, ccb)
            torch.save({"state_dict": head.state_dict(), "horizon": K, "hidden_dim": H,
                        "n_channels": C, "n_codebooks": Q, "epoch": ep, "val_loss": vl}, args.out)
            flag = "  *best*"
        print(f"[ep {ep}] train={tot/len(perm):.3f} val={vl:.3f} | "
              f"head(k1..k{K})={fmt(hacc)} | copy={fmt(cacc)}{flag}")

    print(f"[save] best val_loss={best_val:.3f} -> {args.out}")
    if best_cb is not None:
        hcb, ccb = best_cb
        margin = [h - c for h, c in zip(hcb, ccb)]
        print("\n[per-codebook @ best]  (cb0 = coarse/semantic ... higher cb = finer)")
        print("  codebook:  " + " ".join(f"cb{q}" for q in range(Q)))
        print("  head:      " + fmt(hcb))
        print("  copy:      " + fmt(ccb))
        print("  head-copy: " + " ".join(f"{m:+.3f}" for m in margin))
        print("  -> the cb0 margin is the key number: does the head beat copy on")
        print("     the codebook that carries phonetic/semantic (turn-taking) content?")
    print("Read: head should beat copy in the horizon that hides latency (k2-k3).")


if __name__ == "__main__":
    main()
