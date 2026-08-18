"""v2 evaluation: does LoRA on the last block lift the amendable commit frontier?

Unlike v0/v1, there is no cached feature to load: the adapted backbone must be run LIVE
to produce features. This script:
  1. re-loads the frozen q8 Moshi,
  2. re-injects LoRA into the SAME target modules used in training,
  3. loads the trained adapter weights (load_adapters_into),
  4. runs each held-out conversation live to produce per-frame features,
  5. writes them to a temporary .npz per conversation (same format as Stage B),
  6. hands off to the EXISTING eval_speculative.py so the commit-gate logic is identical
     to v0 --- the only thing that changed is the feature source.

This keeps the comparison honest: v2 vs v0 differ ONLY in the features (adapted vs frozen),
evaluated by the same gates. Compare the amendable frontier, not the loss.

HONEST NOTES:
  * Unvalidated in the assistant sandbox (no GPU/Moshi). Validate on first run.
  * --target-modules MUST match training exactly, or the adapters attach to the wrong
    layers and the eval is meaningless.
  * This does a live forward with grad DISABLED (torch.no_grad), so it is much lighter
    than training --- it should fit easily.
"""
from __future__ import annotations
import argparse, sys, subprocess
from pathlib import Path
import numpy as np


def resolve(obj, dotted):
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def take(h):
    return h[:, 0] if h.dim() == 3 else h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-dir", type=Path, required=True,
                    help="held-out Stage-A token .npy files")
    ap.add_argument("--adapters", type=Path, required=True, help="lora_v2.pt from training")
    ap.add_argument("--head", type=Path, required=True, help="head_lora_v2.pt from training")
    ap.add_argument("--target-modules", required=True,
                    help="SAME comma-separated names used in training")
    ap.add_argument("--feats-out", type=Path, default=Path("feats_v2_eval"),
                    help="dir to write live features (Stage-B format)")
    ap.add_argument("--pairs-out", type=Path, default=Path("pairs_v2_eval"),
                    help="dir to pair features with tokens for eval_speculative")
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers-attr", default="transformer.layers")
    ap.add_argument("--eval-script", type=Path,
                    default=Path("scripts/eval_speculative.py"),
                    help="path to the existing v0 eval_speculative.py")
    ap.add_argument("--save-json", type=Path, default=Path("eval_v2.json"))
    ap.add_argument("--relaxed", default="3:2,4:3,4:2")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    try:
        import torch
        from moshi.models import LMGen, loaders
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from duplex_spec.lora import load_adapters_into
    except ImportError as e:
        sys.exit(f"Import failed ({e}). Ensure moshi + duplex_spec.lora are importable.")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    dev, dt = args.device, getattr(torch, args.dtype)

    # ---- reload frozen q8 backbone ----
    ckpt = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    lm = ckpt.get_moshi(device=dev, dtype=dt)
    try:
        from moshi.utils.quantize import QLinear
        for m in lm.modules():
            if isinstance(m, QLinear):
                m.weight_scb.data = m.weight_scb.data.float()
    except Exception:
        pass
    for p in lm.parameters():
        p.requires_grad_(False)

    # ---- re-inject LoRA into the SAME modules and load trained adapters ----
    final = resolve(lm, args.layers_attr)[-1]
    adapters = load_adapters_into(final, args.adapters, dtype=dt)
    print(f"[lora] re-attached {len(adapters)} adapters from {args.adapters}")
    # sanity: names must match what the user passed
    want = [s.strip() for s in args.target_modules.split(",")]
    got = list(getattr(adapters, "_lora_names", []))
    if set(want) != set(got):
        print(f"[warn] target-modules {want} != saved adapter names {got}; "
              f"features may be wrong if these disagree.")

    needed = lm.num_codebooks - lm.dep_q - 1
    lm_gen = LMGen(lm)

    def frame_tensor(tokens, ch, t):
        return torch.from_numpy(tokens[ch, :, t]).to(dev).long()[None, :, None]

    args.feats_out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.tokens_dir.glob("*.npy"))
    if not files:
        sys.exit(f"No .npy token files in {args.tokens_dir}")

    # ---- live feature extraction through the ADAPTED backbone ----
    for f in files:
        tokens = np.load(f)
        T = tokens.shape[2]
        if args.max_seconds:
            T = min(T, int(args.max_seconds * 12.5))
        feats, frames = [], []
        with torch.no_grad(), lm_gen.streaming(1):
            for t in range(T):
                user = frame_tensor(tokens, 0, t)[:, :needed]
                moshi = frame_tensor(tokens, 1, t)[:, : lm.dep_q]
                out = lm_gen._step(user, depformer_replace_tokens=moshi)
                if out is None:
                    continue
                _tok, hstate = out
                feats.append(take(hstate).float().cpu().numpy())
                frames.append(t)
        feat = np.concatenate(feats, 0)
        np.savez(args.feats_out / f"{f.stem}.npz",
                 feats=feat, frames=np.asarray(frames, np.int64),
                 conv_id=f.stem, layer_spec="lora_v2")
        print(f"[ok] {f.stem}: feats {feat.shape}")

    # ---- pair live features with tokens, then call the EXISTING eval ----
    args.pairs_out.mkdir(parents=True, exist_ok=True)
    for f in files:
        (args.pairs_out / f"{f.stem}.npy").unlink(missing_ok=True)
        (args.pairs_out / f"{f.stem}.npz").unlink(missing_ok=True)
        (args.pairs_out / f"{f.stem}.npy").symlink_to(f.resolve())
        (args.pairs_out / f"{f.stem}.npz").symlink_to((args.feats_out / f"{f.stem}.npz").resolve())

    cmd = [sys.executable, str(args.eval_script),
           "--head", str(args.head), "--pairs-dir", str(args.pairs_out),
           "--device", args.device, "--relaxed", args.relaxed,
           "--save-json", str(args.save_json)]
    print(f"[eval] handing off to: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[done] v2 frontier written to {args.save_json} --- overlay against eval_v0.json "
          f"to see whether LoRA lifted the amendable frontier.")


if __name__ == "__main__":
    main()
