"""One-time fix: wrap the v2 bare-state_dict head into the metadata format that
eval_speculative.py expects (keys: state_dict, head_type, horizon, hidden_dim,
n_channels, n_codebooks). No retraining --- just re-wraps the saved weights.
"""
import argparse, torch
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", type=Path, default=Path("head_lora_v2.pt"))
ap.add_argument("--out", type=Path, default=Path("head_lora_v2_fixed.pt"))
ap.add_argument("--horizon", type=int, default=4)
ap.add_argument("--hidden", type=int, default=4096)
args = ap.parse_args()

obj = torch.load(args.inp, map_location="cpu")
# handle either a bare state_dict or an already-wrapped dict
sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

torch.save({
    "state_dict": sd,
    "head_type": "independent",     # v2 uses the independent head
    "horizon": args.horizon,
    "hidden_dim": args.hidden,
    "n_channels": 2,
    "n_codebooks": 8,
    "val_loss": 3.78,          # label so the eval print shows it's the v2 head
}, args.out)
print(f"[ok] re-saved {args.inp} -> {args.out} with eval metadata "
      f"(horizon={args.horizon}, hidden={args.hidden})")
