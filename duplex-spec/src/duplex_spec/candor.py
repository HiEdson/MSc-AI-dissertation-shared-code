"""CANDOR preprocessing helpers (pure, numpy-only, unit-tested).

The heavy lifting (mp3 decode, resample, Mimi encode) lives in
scripts/candor_preprocess.py; this module holds the logic worth testing in
isolation: mapping stereo channels to speakers, planning frame counts, and
building manifest rows.

CANDOR processed layout (confirmed):
  <conv-uuid>/processed/<conv-uuid>.mp3   stereo, 48kHz, L/R = two speakers
  <conv-uuid>/processed/channel_map.json  {"L": <speakerId>, "R": <speakerId>}
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

MOSHI_SAMPLE_RATE = 24000
MOSHI_FRAME_RATE = 12.5


def order_channels(wav: np.ndarray, channel_map: dict) -> tuple[np.ndarray, list[str]]:
    """Return (wav ordered [L, R], [L_speaker_id, R_speaker_id]).

    wav: stereo array shape [2, n_samples] with row 0 = left, row 1 = right
    (the convention soundfile/librosa use). We keep channel 0 = L, channel 1 = R
    and just report which speaker id each is, so downstream "channel 0" always
    means the left speaker.
    """
    if wav.ndim != 2 or wav.shape[0] != 2:
        raise ValueError(f"expected stereo [2, n], got {wav.shape}")
    if "L" not in channel_map or "R" not in channel_map:
        raise ValueError(f"channel_map needs 'L' and 'R' keys, got {list(channel_map)}")
    return wav, [channel_map["L"], channel_map["R"]]


def expected_frames(n_samples: int,
                    sample_rate: int = MOSHI_SAMPLE_RATE,
                    frame_rate: float = MOSHI_FRAME_RATE) -> int:
    """Number of complete Mimi frames in n_samples at the given rates."""
    frame_size = int(sample_rate / frame_rate)          # 1920 @ 24k/12.5
    return n_samples // frame_size


def resampled_length(n_samples: int, src_sr: int, dst_sr: int = MOSHI_SAMPLE_RATE) -> int:
    """Length after resampling (for planning / assertions)."""
    return int(round(n_samples * dst_sr / src_sr))


@dataclass
class ConversationManifest:
    conv_id: str
    speaker_L: str
    speaker_R: str
    src_sample_rate: int
    n_frames: int
    seconds: float
    token_path: str

    def as_dict(self) -> dict:
        return asdict(self)


def aligned_examples(frames: np.ndarray, T: int, horizon: int) -> list[tuple[int, int]]:
    """Pair each cached feature with a full future-target window.

    frames: [M] conversation-frame index of each cached feature (from Stage B).
    T: total frames in the token sequence (Stage A). horizon: K frames to predict.
    Returns (feature_row, frame_index) for features whose full K-frame future
    exists, i.e. frame_index + horizon < T. Features too close to the end (no
    complete target window) are dropped.
    """
    out = []
    for row, f in enumerate(frames):
        if int(f) + horizon < T:
            out.append((row, int(f)))
    return out


def future_targets(tokens: np.ndarray, frame_index: int, horizon: int) -> np.ndarray:
    """The next `horizon` dual-channel token-pairs after `frame_index`.

    tokens: [2, n_codebooks, T] (from Stage A). Returns [2, n_codebooks, h] for
    frames (frame_index+1 .. frame_index+horizon), clipped at the end of the
    sequence (so the returned h may be < horizon near the tail). This is what a
    feature at `frame_index` is trained to predict.
    """
    start = frame_index + 1
    end = start + horizon
    return tokens[:, :, start:end]


def make_manifest(conv_id: str, speakers: list[str], src_sr: int,
                  n_frames: int, token_path: str,
                  frame_rate: float = MOSHI_FRAME_RATE) -> ConversationManifest:
    return ConversationManifest(
        conv_id=conv_id,
        speaker_L=speakers[0],
        speaker_R=speakers[1],
        src_sample_rate=src_sr,
        n_frames=n_frames,
        seconds=round(n_frames / frame_rate, 2),
        token_path=token_path,
    )
