"""Print a backbone's BackboneConfig + timing facts.

Right now it inspects the mock. Once NTPPBackbone is implemented, point it
there to capture the REAL numbers for your dissertation's setup section:

    python scripts/inspect_backbone.py --backbone ntpp --checkpoint /path/to/ckpt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplex_spec.backbones import MockBackbone  # noqa: E402


def get_config(name: str, checkpoint: str | None):
    if name == "mock":
        return MockBackbone().config
    if name == "ntpp":
        # Config is known from the repo even though loading isn't wired yet.
        from duplex_spec.backbones.ntpp import NTPP_CONFIG
        return NTPP_CONFIG
    if name == "moshi":
        from duplex_spec.backbones.moshi import MOSHI_CONFIG
        return MOSHI_CONFIG
    raise SystemExit(f"unknown backbone: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mock", choices=["mock", "ntpp", "moshi"])
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    cfg = get_config(args.backbone, args.checkpoint)
    print(cfg.summary())
    print(f"  200 ms of lookahead  = {cfg.frames_for_ms(200)} frames")
    print(f"  tokens / second      = {cfg.frame_rate_hz * cfg.n_channels * cfg.n_codebooks:g}")
    print(f"  human turn gap (~200ms) sits at frame index {cfg.frames_for_ms(200)}")


if __name__ == "__main__":
    main()
