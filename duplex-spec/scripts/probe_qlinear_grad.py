"""PROBE (run this FIRST) --- the make-or-break test for v2 LoRA on the q8 backbone.

Path-A LoRA keeps Moshi's q8 weights frozen and trains small adapters placed around
the final block's linear layers. For that to train at all, gradients must flow THROUGH
the frozen q8 `QLinear` layers to reach the adapters --- i.e. QLinear must be
differentiable w.r.t. its INPUT (not its weights). QLinear was built for inference, so
this is not guaranteed. This script answers the single question:

    Does d(output)/d(input) exist and is it non-zero through a q8 QLinear?

If YES  -> Path A is viable; proceed to the LoRA training scaffold.
If NO   -> Path A is dead as-is; adapters must go BETWEEN frozen blocks (around, not
           through, QLinear), which we would design instead.

It also lists the module paths of the final transformer block, so the training script
knows exactly which linear layers to wrap (paths differ across Moshi versions).

Run:
    python scripts/probe_qlinear_grad.py --device cuda
    python scripts/probe_qlinear_grad.py --device cuda --list-final-block

NOTE: unvalidated in the assistant sandbox (no GPU / no Moshi). Read the output
carefully; the VERDICT line is what matters.
"""
from __future__ import annotations
import argparse, sys


def resolve(obj, dotted):
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers-attr", default="transformer.layers",
                    help="dotted path to the temporal transformer layer list")
    ap.add_argument("--list-final-block", action="store_true",
                    help="print every nn.Linear/QLinear in the final block and exit")
    args = ap.parse_args()

    try:
        import torch
        from torch import nn
        from moshi.models import loaders
    except ImportError:
        sys.exit("Need torch + moshi in this env.")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available.")

    ckpt = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    lm = ckpt.get_moshi(device=args.device, dtype=getattr(torch, args.dtype))
    # q8 scale-buffer fix (same as the extraction script)
    try:
        from moshi.utils.quantize import QLinear
        for m in lm.modules():
            if isinstance(m, QLinear):
                m.weight_scb.data = m.weight_scb.data.float()
    except Exception:
        QLinear = None

    layers = resolve(lm, args.layers_attr)
    final = layers[-1]
    print(f"[info] {len(layers)} layers; probing final block: {type(final).__name__}")

    # enumerate candidate linear layers in the final block
    def is_linearish(m):
        if isinstance(m, nn.Linear):
            return True
        if QLinear is not None and isinstance(m, QLinear):
            return True
        return False

    linears = [(name, m) for name, m in final.named_modules() if is_linearish(m)]
    if args.list_final_block or not linears:
        print(f"[final-block linears] {len(linears)} found:")
        for name, m in linears:
            kind = type(m).__name__
            shape = tuple(getattr(m, "weight", torch.empty(0)).shape) if hasattr(m, "weight") else "?"
            print(f"    {name}: {kind} weight={shape}")
        if args.list_final_block:
            sys.exit(0)
        if not linears:
            sys.exit("No linear layers found in final block --- adjust --layers-attr.")

    # pick the first linear-ish layer and test gradient-through-input
    name, layer = linears[0]
    print(f"[probe] testing gradient THROUGH input of: {name} ({type(layer).__name__})")

    # infer input width
    in_features = getattr(layer, "in_features", None)
    if in_features is None and hasattr(layer, "weight"):
        in_features = layer.weight.shape[-1]
    if in_features is None:
        sys.exit("Could not infer in_features; run --list-final-block and inspect.")

    x = torch.randn(1, int(in_features), device=args.device,
                    dtype=getattr(torch, args.dtype), requires_grad=True)
    try:
        with torch.enable_grad():
            y = layer(x)
            loss = y.float().pow(2).sum()
            loss.backward()
    except Exception as e:
        print(f"[error] forward/backward through QLinear raised: {type(e).__name__}: {e}")
        print("VERDICT: NO --- QLinear has no usable backward. Path A (adapters INSIDE the "
              "block) is not viable; adapters must go BETWEEN frozen blocks instead.")
        sys.exit(1)

    g = x.grad
    if g is None:
        print("VERDICT: NO --- input.grad is None. No gradient path through QLinear. "
              "Path A not viable as designed.")
        sys.exit(1)
    gnorm = g.detach().float().norm().item()
    finite = bool(torch.isfinite(g).all())
    print(f"[probe] input.grad norm = {gnorm:.4e}, all-finite = {finite}")
    if gnorm > 0 and finite:
        print("VERDICT: YES --- gradients flow through the q8 QLinear to its input. "
              "Path A is viable: proceed to the LoRA training scaffold. Note the "
              "final-block linear names above --- they are the --target-modules.")
    else:
        print("VERDICT: MARGINAL/NO --- gradient is zero or non-finite. Investigate before "
              "building the training loop (dtype, scale buffers, or QLinear backward).")


if __name__ == "__main__":
    main()
