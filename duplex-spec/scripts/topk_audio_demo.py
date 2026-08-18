"""Top-5 acceptance AUDIO demo --- the perceptual justification for accepting a frame when
the true token is in the head's confident top-5, rather than only when it exactly matches
the argmax.

THE ARGUMENT (and the trap it must avoid)
-----------------------------------------
Your Delta-E analysis showed the top-5 CANDIDATES are acoustically DISTANT from each other,
so the claim is NOT "top-5 tokens sound alike". The claim is:

    when the argmax is 'wrong' by exact match but the TRUE token was in the model's
    confident top-5, the frame decoded with the true token is a perceptually valid
    continuation --- one that exact-match wrongly rejected.

So we find frames where: argmax(cb0) != true(cb0)  AND  true(cb0) in confident top-5, then
decode the FULL frame (all 8 codebooks, both channels) THREE ways and let you listen:

    (1) ground_truth : the real next frame               (reference)
    (2) argmax       : cb0 replaced by the head's argmax (what exact-match commits)
    (3) true_top5    : cb0 = the true token              (what top-5 acceptance permits)

If (3) sounds like (1) and (2) sounds off, top-5 acceptance is recovering perceptually-right
frames that exact-match discards. Decoding full frames (not cb0 alone) is essential --- cb0
in isolation is not listenable.

We decode a short CONTEXT window ending at the frame so the single-frame swap is audible in
motion, not as an 80 ms blip.

Usage:
    PYTHONPATH=src python scripts/topk_audio_demo.py \
        --tokens tokens_eval/<conv>.npy --head head_v0.pt \
        --pairs-dir pairs_eval/ --device cuda \
        --n-examples 6 --context 12 --out demo_audio/
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True, help="Stage-A token .npy [C,Q,T]")
    ap.add_argument("--feats", type=Path, help="matching .npz features; default: sibling of --tokens")
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ent-floor", type=float, default=0.5)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--context", type=int, default=12, help="frames of context before the swap")
    ap.add_argument("--channel", type=int, default=1, help="which channel's cb0 to swap (0/1)")
    ap.add_argument("--out", type=Path, default=Path("demo_audio"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import soundfile as sf
    from moshi.models import loaders
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from duplex_spec.head import MultiStepTPPHead, MultiStepDepHead

    # --- load head ---
    ck = torch.load(args.head, map_location=args.device)
    K = ck["horizon"]
    Head = MultiStepDepHead if ck.get("head_type") == "dep" else MultiStepTPPHead
    head = Head(hidden_dim=ck["hidden_dim"], n_channels=ck["n_channels"],
                n_codebooks=ck["n_codebooks"], codebook_size=2048, horizon=K)
    head.load_state_dict(ck["state_dict"]); head.to(args.device).eval()
    logV = float(np.log(2048.0))

    # --- load Mimi (same loader as Stage A) ---
    print("[load] Mimi ...")
    mckpt = loaders.CheckpointInfo.from_hf_repo(loaders.DEFAULT_REPO)
    mimi = mckpt.get_mimi(device=args.device)

    tokens = np.load(args.tokens)                          # [C, Q, T]
    C, Q, T = tokens.shape
    feats_path = args.feats or args.tokens.with_suffix(".npz")
    d = np.load(feats_path); feats, frames = d["feats"], d["frames"]

    # --- find candidate frames: argmax != true AND true in confident top-5 (chosen channel) ---
    rows = [(int(fr), r) for r, fr in enumerate(frames) if int(fr) + 1 < T]
    fr_idx = np.array([f for f, _ in rows]); ft_row = np.array([r for _, r in rows])
    ch = args.channel
    cand = []                                              # (frame, argmax_tok, true_tok)
    with torch.no_grad():
        for s in range(0, len(ft_row), args.batch):
            br = ft_row[s:s + args.batch]; bf = fr_idx[s:s + args.batch]
            x = torch.from_numpy(feats[br].astype(np.float32)).to(args.device)
            p = torch.softmax(head(x)[:, 0, ch, 0, :], dim=-1)         # [b, V] cb0 horizon1, chan
            ent = (-(p * (p + 1e-12).log()).sum(-1)) / logV
            am = p.argmax(-1)
            for j, f in enumerate(bf):
                true_tok = int(tokens[ch, 0, f + 1])
                pj = p[j]
                ptrue = pj[true_tok]
                rank = int((pj > ptrue).sum())
                if int(am[j]) != true_tok and rank < args.topk and float(ent[j]) < args.ent_floor:
                    cand.append((int(f), int(am[j]), true_tok, rank, float(ent[j])))
    print(f"[find] {len(cand)} frames where argmax!=true but true in confident top-{args.topk}")
    if not cand:
        sys.exit("no qualifying frames; try a different conversation or relax --ent-floor")

    rng = np.random.default_rng(args.seed)
    pick = [cand[i] for i in rng.choice(len(cand), min(args.n_examples, len(cand)), replace=False)]

    args.out.mkdir(parents=True, exist_ok=True)
    sr = mimi.sample_rate

    def decode_window(tok_CQT):
        """tok [C,Q,win] -> mono mix waveform via Mimi (per-channel decode, summed)."""
        mix = None
        with torch.no_grad():
            for c in range(C):
                codes = torch.from_numpy(tok_CQT[c][None]).to(args.device).long()  # [1,Q,win]
                wav = mimi.decode(codes).squeeze().cpu().numpy().astype(np.float32)
                mix = wav if mix is None else mix[:len(wav)] + wav[:len(mix)]
        return mix / max(np.abs(mix).max(), 1e-6)

    manifest = []
    for n, (f, am_tok, true_tok, rank, ent) in enumerate(pick):
        lo = max(0, f - args.context)
        win = tokens[:, :, lo:f + 2].copy()                # context .. frame f+1  [C,Q,win]
        swap_col = (f + 1) - lo                             # index of the swapped frame in win

        gt = win.copy()                                    # (1) ground truth
        v_argmax = win.copy(); v_argmax[ch, 0, swap_col] = am_tok    # (2) argmax commit
        v_true = win.copy();   v_true[ch, 0, swap_col] = true_tok    # (3) top-5 accepted (== gt cb0)

        base = args.out / f"ex{n:02d}_f{f}"
        sf.write(f"{base}_1_ground_truth.wav", decode_window(gt), sr)
        sf.write(f"{base}_2_argmax.wav",        decode_window(v_argmax), sr)
        sf.write(f"{base}_3_true_top5.wav",     decode_window(v_true), sr)
        manifest.append(dict(example=n, frame=f, channel=ch, argmax_token=am_tok,
                             true_token=true_tok, true_rank=rank, entropy=round(ent, 3),
                             files=[f"{base.name}_1_ground_truth.wav",
                                    f"{base.name}_2_argmax.wav",
                                    f"{base.name}_3_true_top5.wav"]))
        print(f"  ex{n:02d} frame {f}: argmax={am_tok} true={true_tok} (rank {rank}, ent {ent:.2f})")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[out] {len(pick)} examples x 3 wavs in {args.out}/")
    print("  Listen in order 1->2->3 per example:")
    print("   1 ground_truth = reference")
    print("   2 argmax       = what exact-match commits (should sound off if argmax was wrong)")
    print("   3 true_top5    = what top-5 acceptance permits (should sound like 1)")
    print("  If 3 ~ 1 and 2 differs, top-5 recovers perceptually-valid frames exact-match rejects.")
    print("  NOTE: report this as qualitative/illustrative; examples are randomly drawn, not")
    print("  cherry-picked, and cb0-swap on a full frame is the honest single-token manipulation.")


if __name__ == "__main__":
    main()
