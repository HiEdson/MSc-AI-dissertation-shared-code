from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_pair(feats_path: Path, tokens_path: Path, horizon: int, target_shift: int = 0):
    """Return (X [N,H], Y [N,C,Q,K], Y0 [N,C,Q]) for one conversation."""
    d = np.load(feats_path)
    feats, frames = d["feats"], d["frames"]
    tokens = np.load(tokens_path)                         # [C, Q, T]
    C, Q, T = tokens.shape
    rows, fs = [], []
    for row, f in enumerate(frames):
        f = int(f)
        start = f + 1 + target_shift
        if start >= 0 and start + horizon <= T:
            rows.append(row); fs.append(f)
    if not rows:
        z = np.zeros((0, feats.shape[1]), np.float32)
        return z, np.zeros((0, C, Q, horizon), np.int64), np.zeros((0, C, Q), np.int64)
    X = np.stack([feats[r] for r in rows]).astype(np.float32)
    Y = np.stack([tokens[:, :, f + 1 + target_shift: f + 1 + target_shift + horizon] for f in fs]).astype(np.int64)
    Y0 = np.stack([tokens[:, :, f] for f in fs]).astype(np.int64)
    return X, Y, Y0


class MoshiHeadDataset(Dataset):
    """Lazy dataset: keeps per-conversation arrays, avoids one giant RAM concat (OOM-safe)."""
    def __init__(self, pairs, horizon, target_shift):
        self.Xs, self.Ys, self.Y0s, self.lookup = [], [], [], []
        for fp, tp in pairs:
            X, Y, Y0 = load_pair(fp, tp, horizon, target_shift)
            if len(X) > 0:
                self.Xs.append(X); self.Ys.append(Y); self.Y0s.append(Y0)
                fi = len(self.Xs) - 1
                self.lookup += [(fi, r) for r in range(len(X))]
            print(f"[data] {fp.name}: {len(X)} examples")

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        f, r = self.lookup[idx]
        return (torch.from_numpy(self.Xs[f][r]).float(),
                torch.from_numpy(self.Ys[f][r]).long(),
                torch.from_numpy(self.Y0s[f][r]).long())



def evaluate(loader, head, dev, K, C, Q):
    from duplex_spec.head_exp2 import tpp_loss
    head.eval()
    tot_loss = n = 0
    eqh_k = torch.zeros(K, device=dev); eqc_k = torch.zeros(K, device=dev)
    eqh_q = torch.zeros(Q, device=dev); eqc_q = torch.zeros(Q, device=dev)
    with torch.no_grad():
        for xb, yb, y0b in loader:
            xb, yb, y0b = xb.to(dev), yb.to(dev), y0b.to(dev)	
            B = xb.shape[0]
            y_tf = yb.permute(0, 3, 1, 2)                 
            
            # ADDED: Pass y0b to the head
            tot_loss += tpp_loss(head(xb, y0=y0b, y=y_tf), yb).item() * B
            n += B
            
            # ADDED: Pass y0b to the head for greedy generation
            pred = head(xb, y0=y0b).argmax(-1)                     
            tgt = y_tf                                     
            copy_pred = y0b[:, None].expand(-1, K, -1, -1)
            eqh = (pred == tgt).float(); eqc = (copy_pred == tgt).float()
            eqh_k += eqh.sum(dim=(0, 2, 3)); eqc_k += eqc.sum(dim=(0, 2, 3))
            eqh_q += eqh.sum(dim=(0, 1, 2)); eqc_q += eqc.sum(dim=(0, 1, 2))
    dk, dq = n * C * Q, n * K * C
    head.train()
    return (tot_loss / n, (eqh_k / dk).tolist(), (eqc_k / dk).tolist(),
            (eqh_q / dq).tolist(), (eqc_q / dq).tolist())


def fmt(xs):
    return " ".join(f"{x:.3f}" for x in xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=Path, action="append", default=[])
    ap.add_argument("--tokens", type=Path, action="append", default=[])
    ap.add_argument("--pairs-dir", type=Path)
    #ap.add_argument("--head", choices=["independent", "dep"], default="independent",help="independent = v0 (codebooks independent); dep = v1 (autoregressive over codebooks)")
    ap.add_argument("--head", choices=["independent", "cascaded"], default="independent",
                    help="independent = v0; cascaded = hybrid fast MLP")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--target-shift", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 (independent), 3e-4 (dep transformer)")
    ap.add_argument("--clip-grad", type=float, default=1.0, help="max grad norm (0 disables); stabilises the dep transformer")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("head.pt"))
    ap.add_argument("--save-json", type=Path, help="train summary for plot_results.py")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from duplex_spec.head_exp2 import MultiStepTPPHead, MultiStepCascadedMLPHead, tpp_loss

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
    print(f"[data] total {len(dataset)} examples  H={H} C={C} Q={Q} K={K} "
          f"target_shift={args.target_shift}  head={args.head}")

    n_val = max(1, int(len(dataset) * args.val_frac)); n_train = len(dataset) - n_val
    torch.manual_seed(0)
    tr, va = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(va, batch_size=args.batch, shuffle=False)

    dev = args.device
    
    if args.head == "cascaded":
        head = MultiStepCascadedMLPHead(
            hidden_dim=H, 
            n_channels=C, 
            n_codebooks=Q, 
            codebook_size=2048, 
            horizon=K,
            trunk_dim=2048
        ).to(dev)
    else:
        head = MultiStepTPPHead(
            hidden_dim=H, 
            n_channels=C, 
            n_codebooks=Q, 
            codebook_size=2048, 
            horizon=K,
            trunk_dim=2048
        ).to(dev)
    
    
    lr = args.lr if args.lr is not None else (3e-4 if args.head == "dep" else 1e-3)
    opt = (torch.optim.AdamW(head.parameters(), lr=lr) if args.head == "dep"
           else torch.optim.Adam(head.parameters(), lr=lr))
    print(f"[head] type={args.head} params={sum(p.numel() for p in head.parameters())/1e6:.1f}M "
          f"lr={lr:g} clip={args.clip_grad}")

    #added 1 ---------------
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = 5
    total_iters = args.epochs * len(train_loader)
    warmup_iters = warmup_epochs * len(train_loader)
    
    warmup = LinearLR(opt, start_factor=0.05, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(opt, T_max=(total_iters - warmup_iters), eta_min=1e-5)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_iters])
    #close added 1 ---------
    
    best_val = float("inf"); best = None
    for ep in range(args.epochs):
        tot = ntr = 0
        for xb, yb, y0b in train_loader:
            xb, yb, y0b = xb.to(dev), yb.to(dev), y0b.to(dev)
            opt.zero_grad()
            loss = tpp_loss(head(xb, y0=y0b, y=yb.permute(0, 3, 1, 2)), yb)
            #loss = tpp_loss(head(xb, yb.permute(0, 3, 1, 2)), yb)   # teacher-forced (dep); y ignored (v0)
            if not torch.isfinite(loss):
                continue                                            # skip a bad batch, don't poison the weights
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.clip_grad)
            opt.step()
            
            #added 2 ---------------
            scheduler.step()
            #end of added 2 ---------------
            
            tot += loss.item() * len(xb); ntr += len(xb)
        vl, hacc, cacc, hcb, ccb = evaluate(val_loader, head, dev, K, C, Q)
        
        # Get current LR for logging
        current_lr = opt.param_groups[0]['lr']
        
        flag = ""
        if vl < best_val:
            best_val = vl; best = (hacc, cacc, hcb, ccb)
            torch.save({"state_dict": head.state_dict(), "head_type": args.head, "horizon": K,
                        "hidden_dim": H, "n_channels": C, "n_codebooks": Q, "epoch": ep, "val_loss": vl}, args.out)
            flag = "   *best*"
        print(f"[ep {ep}] lr={current_lr:.2e} train={tot/ntr:.3f} val={vl:.3f} | head(k1..k{K})={fmt(hacc)} | copy={fmt(cacc)}{flag}")

    print(f"[save] best val_loss={best_val:.3f} -> {args.out}")
    if best is not None:
        hacc, cacc, hcb, ccb = best
        margin = [h - c for h, c in zip(hcb, ccb)]
        print("\n[per-codebook @ best]  (cb0 = coarse/semantic ... higher cb = finer)")
        print("  codebook:  " + " ".join(f"cb{q}" for q in range(Q)))
        print("  head:      " + fmt(hcb)); print("  copy:      " + fmt(ccb))
        print("  head-copy: " + " ".join(f"{m:+.3f}" for m in margin))
        if args.save_json:
            args.save_json.write_text(json.dumps({
                "meta": {"head_type": args.head, "val_loss": best_val, "horizon": K},
                "per_horizon": {"head": hacc, "copy": cacc},
                "per_codebook": {"head": hcb, "copy": ccb}}, indent=1))
            print(f"[json] {args.save_json}  (-> python scripts/plot_results.py --train-json {args.save_json} --out figs/)")
    print("Read: head should beat copy in the horizon that hides latency (k2-k3).")


if __name__ == "__main__":
    main()
