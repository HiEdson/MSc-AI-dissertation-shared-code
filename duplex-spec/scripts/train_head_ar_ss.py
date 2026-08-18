"""Standalone trainer for the horizon-autoregressive head (v3).

Kept separate from train_head.py so the working v0/dep pipeline is untouched. It reuses
that module's dataset, evaluation and loss verbatim --- only the head class differs --- so
the training recipe, teacher-forced val loss, and greedy accuracy readout are identical to
v0. That parity is the point: any change in the numbers is attributable to the head, not the
harness.

The greedy-accuracy line is the exposure-bias watch: val loss is teacher-forced, accuracy is
greedy rollout. If val drops below v0's ~4.046 but greedy far-horizon accuracy (k2..k4) does
NOT improve, the head is leaning on teacher frames it will not have at inference --- the same
trap the depformer fell into. Watch head(k2..k4) vs v0, not just val.

Usage (match your v0 recipe; only the script changes):
    PYTHONPATH=src python scripts/train_head_ar.py \
        --pairs-dir pairs/ --epochs 40 --out head_ar.pt --device cuda
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", type=Path)
    ap.add_argument("--feats", type=Path, action="append", default=[])
    ap.add_argument("--tokens", type=Path, action="append", default=[])
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--target-shift", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("head_ar.pt"))
    ap.add_argument("--save-json", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ss-start", type=float, default=1.0,
                    help="teacher-forcing prob at epoch 0 (1.0 = full teacher forcing)")
    ap.add_argument("--ss-end", type=float, default=0.3,
                    help="teacher-forcing prob floor (reached by --ss-anneal-epochs)")
    ap.add_argument("--ss-anneal-epochs", type=int, default=None,
                    help="epochs to linearly anneal ss-start->ss-end (default: all epochs)")
    args = ap.parse_args()

    # reuse the exact harness from train_head.py (dataset / eval / loss / formatting)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_head import MoshiHeadDataset, evaluate, fmt
    from duplex_spec.head import tpp_loss
    try:
        from duplex_spec_v2.head_ar import HorizonARHead
    except ImportError:
        from duplex_spec.head_ar import HorizonARHead

    pairs = list(zip(args.feats, args.tokens))
    if args.pairs_dir:
        for npz in sorted(args.pairs_dir.glob("*.npz")):
            npy = npz.with_suffix(".npy")
            if npy.exists():
                pairs.append((npz, npy))
    if not pairs:
        sys.exit("No (feats, tokens) pairs. Use --feats/--tokens or --pairs-dir.")

    dataset = MoshiHeadDataset(pairs, args.horizon, args.target_shift)
    if len(dataset) == 0:
        sys.exit("No training examples found.")
    sx, sy, _ = dataset[0]
    H = sx.shape[0]; C, Q, K = sy.shape[0], sy.shape[1], sy.shape[2]
    print(f"[data] total {len(dataset)} examples  H={H} C={C} Q={Q} K={K}  head=ar")

    n_val = max(1, int(len(dataset) * args.val_frac)); n_train = len(dataset) - n_val
    torch.manual_seed(0)                                   # same seed/split as v0
    tr, va = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(va, batch_size=args.batch, shuffle=False)

    dev = args.device
    head = HorizonARHead(hidden_dim=H, n_channels=C, n_codebooks=Q,
                         codebook_size=2048, horizon=K).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    print(f"[head] type=ar params={sum(p.numel() for p in head.parameters())/1e6:.1f}M "
          f"lr={args.lr:g} clip={args.clip_grad}")

    anneal = args.ss_anneal_epochs or args.epochs
    best_val = float("inf"); best = None
    for ep in range(args.epochs):
        frac = min(1.0, ep / max(anneal - 1, 1))
        tf_prob = args.ss_start + (args.ss_end - args.ss_start) * frac   # linear anneal
        head.train(); tot = ntr = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            # scheduled sampling: teacher frame w.p. tf_prob, else head's own greedy pred
            loss = tpp_loss(head(xb, yb.permute(0, 3, 1, 2), tf_prob=tf_prob), yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.clip_grad)
            opt.step()
            tot += loss.item() * len(xb); ntr += len(xb)
        vl, hacc, cacc, hcb, ccb = evaluate(val_loader, head, dev, K, C, Q)
        flag = ""
        if vl < best_val:
            best_val = vl; best = (hacc, cacc, hcb, ccb)
            torch.save({"state_dict": head.state_dict(), "head_type": "ar", "horizon": K,
                        "hidden_dim": H, "n_channels": C, "n_codebooks": Q,
                        "epoch": ep, "val_loss": vl}, args.out)
            flag = "   *best*"
        print(f"[ep {ep}] tf={tf_prob:.2f} train={tot/ntr:.3f} val={vl:.3f} | "
              f"head(k1..k{K})={fmt(hacc)} | copy={fmt(cacc)}{flag}")

    print(f"[save] best val_loss={best_val:.3f} -> {args.out}")
    if best is not None:
        hacc, cacc, hcb, ccb = best
        margin = [h - c for h, c in zip(hcb, ccb)]
        print("\n[per-codebook @ best]")
        print("  head:      " + fmt(hcb)); print("  copy:      " + fmt(ccb))
        print("  head-copy: " + " ".join(f"{m:+.3f}" for m in margin))
        if args.save_json:
            args.save_json.write_text(json.dumps({
                "meta": {"head_type": "ar", "val_loss": best_val, "horizon": K},
                "per_horizon": {"head": hacc, "copy": cacc}}, indent=2))
            print(f"[json] {args.save_json}")


if __name__ == "__main__":
    main()
