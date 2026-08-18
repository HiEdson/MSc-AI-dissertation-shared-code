"""Greedy per-horizon accuracy of a trained head on held-out features.

The AR head's training log reports TEACHER-FORCED val loss, which flatters exposure-biased
checkpoints. This script measures the honest number: GREEDY rollout accuracy (head(x) with no
teacher frames), per horizon and per codebook, exactly as inference will run it. Compares
against the copy/persistence baseline.

Decision: if greedy k3/k4 recover toward v0's (~0.287 / 0.280), scheduled sampling worked and
the head is worth pushing through the JS gate + probe. If k3/k4 stay collapsed (~0.11), the
exposure bias is structural and the AR direction is closed.

Works for any head_type in the checkpoint (ar / dep / v0), so you can run the same command on
head_v0.pt to print the reference row.

Usage:
    PYTHONPATH=src python scripts/greedy_acc.py --head head_ar_ss.pt --pairs-dir pairs_eval/ --device cuda
    PYTHONPATH=src python scripts/greedy_acc.py --head head_v0.pt    --pairs-dir pairs_eval/ --device cuda
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
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from duplex_spec.head import MultiStepTPPHead, MultiStepDepHead

    ck = torch.load(args.head, map_location=args.device)
    htype = ck.get("head_type", "v0")
    K = ck["horizon"]; C = ck["n_channels"]; Q = ck["n_codebooks"]
    if htype == "ar":
        try:
            from duplex_spec_v2.head_ar import HorizonARHead
        except ImportError:
            from duplex_spec.head_ar import HorizonARHead
        head = HorizonARHead(hidden_dim=ck["hidden_dim"], n_channels=C, n_codebooks=Q,
                             codebook_size=2048, horizon=K)
    elif htype == "dep":
        head = MultiStepDepHead(hidden_dim=ck["hidden_dim"], n_channels=C, n_codebooks=Q,
                                codebook_size=2048, horizon=K)
    else:
        head = MultiStepTPPHead(hidden_dim=ck["hidden_dim"], n_channels=C, n_codebooks=Q,
                                codebook_size=2048, horizon=K)
    head.load_state_dict(ck["state_dict"]); head.to(args.device).eval()
    print(f"[head] {args.head.name}  type={htype}  K={K}  (val_loss@save={ck.get('val_loss')})")

    pairs = [(z, z.with_suffix(".npy")) for z in sorted(args.pairs_dir.glob("*.npz"))
             if z.with_suffix(".npy").exists()]
    if not pairs:
        sys.exit("no pairs")

    # accumulate correct/total per horizon (head greedy vs truth; copy vs truth), all codebooks
    hit_h = np.zeros(K); hit_c = np.zeros(K); tot = np.zeros(K)
    # per-codebook (cb) accuracy at each horizon, head only
    hit_cb = np.zeros((K, Q)); tot_cb = np.zeros((K, Q))
    with torch.no_grad():
        for npz, npy in pairs:
            d = np.load(npz); feats, frames = d["feats"], d["frames"]
            tokens = np.load(npy); Tlen = tokens.shape[2]
            rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + K < Tlen]
            if not rows:
                continue
            fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])
            for s in range(0, len(ft_row), args.batch):
                br = ft_row[s:s + args.batch]; bf = fr_idx[s:s + args.batch]
                x = torch.from_numpy(feats[br].astype(np.float32)).to(args.device)
                pred = head(x).argmax(-1).cpu().numpy()          # [b,K,C,Q] greedy
                # targets [b,K,C,Q]
                tgt = np.stack([tokens[:, :, f + 1:f + 1 + K] for f in bf]).transpose(0, 3, 1, 2)
                # copy baseline: repeat current frame
                cur = np.stack([tokens[:, :, f] for f in bf])[:, None, :, :]  # [b,1,C,Q]
                cur = np.repeat(cur, K, axis=1)                              # [b,K,C,Q]
                for k in range(K):
                    hit_h[k] += (pred[:, k] == tgt[:, k]).sum()
                    hit_c[k] += (cur[:, k] == tgt[:, k]).sum()
                    tot[k] += tgt[:, k].size
                    for q in range(Q):
                        hit_cb[k, q] += (pred[:, k, :, q] == tgt[:, k, :, q]).sum()
                        tot_cb[k, q] += tgt[:, k, :, q].size

    acc_h = hit_h / np.maximum(tot, 1); acc_c = hit_c / np.maximum(tot, 1)
    print("\n  GREEDY per-horizon accuracy (all codebooks):")
    print("    horizon:  " + " ".join(f"k{k+1}" for k in range(K)))
    print("    head:     " + " ".join(f"{a:.3f}" for a in acc_h))
    print("    copy:     " + " ".join(f"{a:.3f}" for a in acc_c))
    print("    head-copy:" + " ".join(f"{h-c:+.3f}" for h, c in zip(acc_h, acc_c)))
    print("\n  (v0 reference greedy: k1 .300  k2 .293  k3 .287  k4 .280)")
    print("  If k3/k4 are near v0, scheduled sampling recovered the far horizons.")
    print("  If k3/k4 << v0 (e.g. ~.11), exposure bias persists -> AR direction closed.\n")

    acc_cb = hit_cb / np.maximum(tot_cb, 1)
    print("  GREEDY per-codebook accuracy (cb0 = coarse/semantic):")
    print("    " + "  ".join(f"cb{q}" for q in range(Q)))
    for k in range(K):
        print(f"    k{k+1}: " + " ".join(f"{acc_cb[k,q]:.3f}" for q in range(Q)))


if __name__ == "__main__":
    main()
