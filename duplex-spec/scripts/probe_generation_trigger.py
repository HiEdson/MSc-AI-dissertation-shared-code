"""FEASIBILITY PROBE for option-1 (timing-trigger) handoff control.

Question this answers, before any demo is built: can we control WHEN Moshi begins
producing speech, using the streaming _step interface? Option 1 does not inject predicted
content --- it only decides the moment Moshi's own generation starts, letting Moshi produce
the actual tokens. So we need one thing to be true:

    feeding Moshi silence/listening tokens on its OWN channel keeps it quiet, and
    letting it generate freely makes it speak --- switchable per frame.

If true, the handoff predictor's fire = the switch, and the demo is tractable.

This runs Moshi on one real conversation's USER channel for a few hundred frames, in two
regimes, and reports how much Moshi's own output channel is 'active' (non-silence) in each:
  A) FREE:   let Moshi generate its own frames every step (normal).
  B) GATED:  force Moshi's own channel to silence for the first half, free for the second.

If B shows near-silence in the first half and activity in the second, the trigger works and
option 1 is feasible. If not, we learn the switch is elsewhere and adjust before building.

Unvalidated in the assistant sandbox (no GPU / Moshi). Run on your machine:
    PYTHONPATH=src python scripts/probe_generation_trigger.py \
        --tokens tokens_eval/<conv>.npy --device cuda --frames 400
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True, help="a Stage-A token .npy [C,Q,T]")
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--frames", type=int, default=400)
    args = ap.parse_args()

    import torch
    from moshi.models import LMGen, loaders

    ckpt = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    lm = ckpt.get_moshi(device=args.device, dtype=getattr(torch, args.dtype))
    try:
        from moshi.utils.quantize import QLinear
        for m in lm.modules():
            if isinstance(m, QLinear):
                m.weight_scb.data = m.weight_scb.data.float()
    except Exception:
        pass

    tokens = np.load(args.tokens)                      # [C, Q, T]
    C, Q, T = tokens.shape
    T = min(T, args.frames)
    needed = lm.num_codebooks - lm.dep_q - 1
    dev = args.device

    # infer silence token per channel (modal cb0) so we can force "listening"
    sil = [int(np.bincount(tokens[c, 0, :]).argmax()) for c in range(C)]
    print(f"[probe] silence cb0 per channel: {sil}")

    def user_frame(t):
        return torch.from_numpy(tokens[0, :, t]).to(dev).long()[None, :, None][:, :needed]

    def silent_moshi_frame():
        # a 'listening' frame for Moshi's own channel: silence token across its dep codebooks
        f = torch.full((1, lm.dep_q, 1), sil[1], dtype=torch.long, device=dev)
        return f

    def moshi_cb0(gen):
        # gen = out[0], shape [1, 9, 1] = [text, cb0..cb7]; audio cb0 is index 1
        return int(gen[0, 1, 0].item())

    def activity_of(gen):
        # non-silence on Moshi's own generated cb0
        return int(moshi_cb0(gen) != sil[1]) if gen is not None else 0

    def run(gate_first_half):
        lm_gen = LMGen(lm)
        acts = []
        with torch.no_grad(), lm_gen.streaming(1):
            for t in range(T):
                u = user_frame(t)
                if gate_first_half and t < T // 2:
                    # force Moshi to 'listen' (silence on its channel) -> should stay quiet
                    out = lm_gen._step(u, depformer_replace_tokens=silent_moshi_frame())
                else:
                    # let Moshi generate its own channel freely
                    out = lm_gen._step(u)
                if out is None:
                    continue
                gen = out[0] if isinstance(out, tuple) else out          # [1,9,1] generated frame
                acts.append(activity_of(gen))
        return np.array(acts)

    print("[A] FREE regime (Moshi generates every frame):")
    a = run(gate_first_half=False)
    print(f"    activity: first-half={a[:len(a)//2].mean():.2f}  second-half={a[len(a)//2:].mean():.2f}")

    print("[B] GATED regime (forced silent first half, free second half):")
    b = run(gate_first_half=True)
    fh, sh = b[:len(b)//2].mean(), b[len(b)//2:].mean()
    print(f"    activity: first-half={fh:.2f}  second-half={sh:.2f}")

    print("\nVERDICT:")
    if fh < 0.1 and sh > fh + 0.15:
        print("  YES -- forcing silence keeps Moshi quiet, releasing lets it speak.")
        print("  Option-1 timing trigger is feasible: gate Moshi's channel on the handoff signal.")
    else:
        print("  UNCLEAR -- gating did not cleanly withhold/release speech. The trigger point")
        print("  is elsewhere (or the output token layout differs). Inspect `out` structure and")
        print("  how Moshi's own channel is represented before building the demo.")
        print(f"  (debug: FREE fh={a[:len(a)//2].mean():.2f}/sh={a[len(a)//2:].mean():.2f}, "
              f"GATED fh={fh:.2f}/sh={sh:.2f})")


if __name__ == "__main__":
    main()
