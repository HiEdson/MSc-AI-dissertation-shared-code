"""CANDOR -> dual-channel Mimi tokens (Stage A of the data pipeline).

Turns one processed CANDOR conversation into a dual-channel Mimi token tensor
and a manifest row. These tokens are BOTH the model input AND the training
labels (the real future token-pairs the speculative head learns to predict).
Moshi hidden-state extraction is Stage B (separate, uses the frozen LM).

Run where moshi is installed (Mimi is ~96M, fits easily; GPU optional):
    pip install moshi librosa soundfile
    python candor_preprocess.py --conv-dir /path/to/<conv-uuid> --out-dir tokens/
    python candor_preprocess.py --root /path/to/candor_extracted --out-dir tokens/   # batch

Output per conversation:
    tokens/<conv-uuid>.npy        int64 array [2, n_codebooks, T]  (ch0=L, ch1=R)
    tokens/manifest.jsonl         one JSON row per conversation

NOTE: written against documented APIs but not executed in the assistant's
sandbox (no GPU/codec). Validate the first conversation's output shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from duplex_spec.candor import (  # noqa: E402
    MOSHI_SAMPLE_RATE,
    expected_frames,
    make_manifest,
    order_channels,
)


def find_conversation(conv_dir: Path) -> tuple[Path, dict]:
    """Locate the stereo mp3 and channel_map.json inside a conversation folder."""
    processed = conv_dir / "processed"
    if not processed.is_dir():
        processed = conv_dir  # allow pointing straight at processed/
    cmap_path = processed / "channel_map.json"
    if not cmap_path.exists():
        raise FileNotFoundError(f"no channel_map.json under {processed}")
    channel_map = json.loads(cmap_path.read_text())
    mp3s = [p for p in processed.glob("*.mp3")]
    if not mp3s:
        raise FileNotFoundError(f"no .mp3 under {processed}")
    # The conversation-level stereo file is named <conv-uuid>.mp3 (UUID with dashes).
    conv_mp3 = max(mp3s, key=lambda p: p.stat().st_size)  # the full mix is the largest
    return conv_mp3, channel_map


def load_stereo_24k(mp3_path: Path) -> tuple[np.ndarray, int]:
    """Decode mp3 -> stereo float32 [2, n], remember source sr, resample to 24k."""
    import librosa
    # sr=None keeps native rate; mono=False keeps both channels -> shape [2, n]
    wav, src_sr = librosa.load(str(mp3_path), sr=None, mono=False)
    if wav.ndim == 1:
        raise ValueError("expected stereo audio but got mono — check the file/channel_map")
    wav24 = librosa.resample(wav, orig_sr=src_sr, target_sr=MOSHI_SAMPLE_RATE)
    return wav24.astype(np.float32), int(src_sr)


"""def encode_dual_channel(mimi, wav24: np.ndarray) -> np.ndarray:
    ""Mimi-encode each speaker channel -> dual-channel tokens [2, n_codebooks, T].""
    import torch
    device = next(mimi.parameters()).device
    per_channel = []
    with torch.no_grad():
        for ch in range(2):
            mono = torch.from_numpy(wav24[ch]).to(device)[None, None]  # [1, 1, n]
            codes = mimi.encode(mono)                                  # [1, n_codebooks, T]
            per_channel.append(codes[0].cpu().numpy())
    T = min(c.shape[1] for c in per_channel)                           # align lengths
    return np.stack([c[:, :T] for c in per_channel], axis=0).astype(np.int64)  # [2, K, T]"""

def encode_dual_channel(mimi, wav24: np.ndarray, chunk_seconds: float = 20.0) -> np.ndarray:
    """Mimi-encode each speaker channel in time-chunks -> [2, n_codebooks, T]."""
    import torch
    device = next(mimi.parameters()).device
    frame_size = int(mimi.sample_rate / mimi.frame_rate)        # 1920 @ 24k/12.5
    # round the chunk to a whole number of frames so nothing is dropped at seams
    chunk_samples = int(chunk_seconds * mimi.sample_rate)
    chunk_samples -= chunk_samples % frame_size

    per_channel = []
    with torch.no_grad():
        for ch in range(2):
            mono = torch.from_numpy(wav24[ch]).to(device)
            pieces = []
            for start in range(0, mono.shape[0] - frame_size + 1, chunk_samples):
                seg = mono[start:start + chunk_samples][None, None]   # [1,1,n]
                codes = mimi.encode(seg)                              # [1,K,t]
                pieces.append(codes[0].cpu().numpy())
                del codes
            per_channel.append(np.concatenate(pieces, axis=1))       # [K, T]
            torch.cuda.empty_cache()
    T = min(c.shape[1] for c in per_channel)
    return np.stack([c[:, :T] for c in per_channel], axis=0).astype(np.int64)


def process_one(conv_dir: Path, out_dir: Path, mimi) -> dict:
    conv_mp3, channel_map = find_conversation(conv_dir)
    conv_id = conv_mp3.stem
    wav24, src_sr = load_stereo_24k(conv_mp3)
    wav24, speakers = order_channels(wav24, channel_map)
    tokens = encode_dual_channel(mimi, wav24)                          # [2, K, T]
    token_path = out_dir / f"{conv_id}.npy"
    np.save(token_path, tokens)
    man = make_manifest(conv_id, speakers, src_sr, n_frames=tokens.shape[2],
                        token_path=str(token_path))
    print(f"[ok] {conv_id}: tokens {tuple(tokens.shape)}  "
          f"({man.seconds:.1f}s, src {src_sr}Hz, expected_frames~"
          f"{expected_frames(wav24.shape[1])})")
    return man.as_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--conv-dir", type=Path, help="one conversation folder")
    g.add_argument("--root", type=Path, help="folder of many conversation folders")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="cap #conversations (for a first batch)")
    args = ap.parse_args()

    try:
        from moshi.models import loaders
    except ImportError:
        sys.exit("Need moshi:  pip install moshi librosa soundfile")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] Mimi on {args.device} ...")
    ckpt = loaders.CheckpointInfo.from_hf_repo(loaders.DEFAULT_REPO)
    mimi = ckpt.get_mimi(device=args.device)

    if args.conv_dir:
        convs = [args.conv_dir]
    else:
        convs = sorted(p for p in args.root.iterdir() if p.is_dir())
    if args.limit:
        convs = convs[: args.limit]
    print(f"[plan] {len(convs)} conversation(s) -> {args.out_dir}")

    manifest_path = args.out_dir / "manifest.jsonl"
    n_ok = 0
    with manifest_path.open("w") as mf:
        for conv in convs:
            try:
                row = process_one(conv, args.out_dir, mimi)
                mf.write(json.dumps(row) + "\n")
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {conv.name}: {type(e).__name__}: {e}")
    print(f"[done] {n_ok}/{len(convs)} ok  -> {manifest_path}")


if __name__ == "__main__":
    main()
