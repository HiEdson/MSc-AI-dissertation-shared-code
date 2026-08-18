"""Hand-rolled LoRA for Path-A v2, via FORWARD HOOKS (not module replacement).

Why hooks: Moshi's streaming transformer type-checks its attention projections
(e.g. transformer.py raises "Unknown type ... for linear" if in_proj is not a
QLinear/nn.Linear). Replacing a layer with a wrapper object breaks that check.
Instead we LEAVE the frozen QLinear in place and register a forward hook that adds
the low-rank update to its output:  y <- y + (alpha/r) * B(A(x)).
Moshi still sees a QLinear; the adapter still trains. Gradients flow through the
frozen QLinear to its input (verified by the probe), and into A/B via the hook.
"""
from __future__ import annotations
import math
from typing import Iterable, List
import torch
from torch import nn


class LoRAAdapter(nn.Module):
    """Low-rank update attached to a base layer via a forward hook (base unchanged)."""

    def __init__(self, in_features: int, out_features: int, r: int = 8,
                 alpha: int = 16, dropout: float = 0.0,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.A = nn.Parameter(torch.zeros(r, in_features, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(out_features, r, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)                       # B=0 -> starts as identity
        self._handle = None

    def _hook(self, module, inputs, output):
        x = inputs[0]                                # input to the frozen base layer
        A = self.A.to(x.dtype)                       # cast adapter to the input's dtype
        B = self.B.to(x.dtype)
        lora = torch.nn.functional.linear(self.drop(x), A)   # [*, r]
        lora = torch.nn.functional.linear(lora, B)           # [*, out]
        return output + self.scaling * lora.to(output.dtype)

    def attach(self, base: nn.Module):
        self._handle = base.register_forward_hook(self._hook)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove(); self._handle = None


def _infer_dims(layer: nn.Module):
    inf = getattr(layer, "in_features", None)
    outf = getattr(layer, "out_features", None)
    if inf is None or outf is None:
        w = getattr(layer, "weight", None)
        if w is not None and w.dim() == 2:
            outf, inf = w.shape
    return inf, outf


def inject_lora(final_block: nn.Module, target_names: Iterable[str],
                r: int = 8, alpha: int = 16, dropout: float = 0.0,
                dtype: torch.dtype = torch.bfloat16) -> "nn.ModuleList":
    """Attach a LoRAAdapter (via hook) to each named submodule of `final_block`.
    Returns an nn.ModuleList of the adapters so the optimiser and save/load can find
    them. The base layers are left untouched (Moshi's type checks still pass).
    """
    adapters = nn.ModuleList()
    adapters._lora_names = []          # parallel list of the wrapped module names
    for name in dict.fromkeys(target_names):          # dedup, keep order
        parts = name.split(".")
        parent = final_block
        for p in parts[:-1]:
            parent = getattr(parent, p)
        base = getattr(parent, parts[-1])
        inf, outf = _infer_dims(base)
        if inf is None or outf is None:
            print(f"[lora] skip {name}: could not infer dims"); continue
        ad = LoRAAdapter(inf, outf, r=r, alpha=alpha, dropout=dropout, dtype=dtype)
        ad.attach(base).to(next(base.parameters()).device)
        adapters.append(ad); adapters._lora_names.append(name)
        print(f"[lora] hooked {name}: in={inf} out={outf} r={r} alpha={alpha}")
    return adapters


def lora_parameters(adapters: "nn.ModuleList"):
    for ad in adapters:
        yield ad.A
        yield ad.B


def save_adapters(adapters: "nn.ModuleList", path):
    """Save adapters keyed by their wrapped-module name, so eval can re-attach exactly."""
    blob = {name: {"A": ad.A.detach().cpu(), "B": ad.B.detach().cpu(),
                   "r": ad.r, "scaling": ad.scaling}
            for name, ad in zip(adapters._lora_names, adapters)}
    torch.save(blob, path)


def load_adapters_into(final_block: nn.Module, path, dtype=torch.bfloat16) -> "nn.ModuleList":
    """Recreate + attach adapters from a saved blob (for eval). Names must match training."""
    blob = torch.load(path, map_location="cpu")
    adapters = nn.ModuleList(); adapters._lora_names = []
    for name, d in blob.items():
        parts = name.split("."); parent = final_block
        for p in parts[:-1]:
            parent = getattr(parent, p)
        base = getattr(parent, parts[-1])
        outf, inf = d["B"].shape[0], d["A"].shape[1]
        ad = LoRAAdapter(inf, outf, r=d["r"], alpha=int(d["scaling"] * d["r"]), dtype=dtype)
        ad.A.data = d["A"].to(dtype); ad.B.data = d["B"].to(dtype)
        ad.attach(base).to(next(base.parameters()).device)
        adapters.append(ad); adapters._lora_names.append(name)
    return adapters
