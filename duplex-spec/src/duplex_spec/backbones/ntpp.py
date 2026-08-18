"""NTPP backbone adapter -- config confirmed from the repo, loading still TODO.

Real specs below are from higgs_audio_inference/INFERENCE_GUIDE.md (not guessed):
frame rate 50 Hz, 8 codebooks, codebook size 1024, 16 kHz, dual-channel.

ENTRYPOINTS (there is NO inference.py):
  higgs_audio_inference/infer_dual_channel.py   <- turn-taking (ch1 | ch0)
  higgs_audio_inference/infer_single_channel.py
  helpers: load_model(), prepare_input_batch(), generate_channel1(), decode_to_audio()

FEASIBILITY (confirmed from scripts/continue_pretrain/llama.sh):
  Released training = Llama-3-8B, torchrun nproc_per_node=8, FSDP full_shard.
  -> cannot train on one 16GB GPU. Use the Phi backbone (ntpp/model/dual_phi.py,
     ~1.3B) for the trainable head; treat the 8B/Higgs path as inference baseline.

The audio backbone at inference is Boson's HiggsAudioModel + bosonai/
higgs-audio-v2-tokenizer (multi-GB) -- measure VRAM before relying on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backbone import StepOutput
from ..config import BackboneConfig

# Confirmed from the repo's inference guide.
NTPP_CONFIG = BackboneConfig(
    name="ntpp-higgs",
    frame_rate_hz=50.0,        # 1 frame = 20 ms; human ~200ms gap = 10 frames
    sample_rate=16000,
    n_channels=2,
    n_codebooks=8,
    codebook_size=1024,
    hidden_dim=-1,             # TODO: set from the chosen LLM (Phi ~2048, Llama-3-8B 4096)
    has_text_stream=True,      # ChatML carries text tokens alongside audio
)


class NTPPBackbone:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        # TODO: mirror infer_dual_channel.load_model(): build HiggsAudioModel /
        # dual_phi from checkpoint, load tokenizer, set hidden_dim on a copy of
        # NTPP_CONFIG, wire step()/encode()/decode() to their collator + generate.
        raise NotImplementedError(
            "Config is known (NTPP_CONFIG); model loading still needs wiring to "
            "higgs_audio_inference.infer_dual_channel helpers. Use MockBackbone meanwhile."
        )

    @property
    def config(self) -> BackboneConfig:
        return NTPP_CONFIG

    def reset(self) -> Any: ...
    def encode(self, wav: np.ndarray) -> np.ndarray: ...
    def decode(self, tokens: np.ndarray) -> np.ndarray: ...
    def step(self, input_pair: np.ndarray, state: Any) -> tuple[StepOutput, Any]: ...
