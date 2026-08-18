#!/usr/bin/env bash
# process_split.sh — run Stage A (tokens) + Stage B (features) for one split,
# then pair the outputs with symlinks.
#
# Usage:
#   ./process_split.sh <conversations_dir> <split_name>
#
# Examples:
#   ./process_split.sh /path/to/candor_train train
#   ./process_split.sh /path/to/candor_eval  eval
#
# Produces (for split=train):  tokens/  feats/  pairs/
#          (for split=eval):   tokens_eval/  feats_eval/  pairs_eval/
#
# Re-runnable: skips any conversation whose output already exists, so you can
# stop/restart safely. Run inside the project root, with moshi-venv activated.

set -euo pipefail

# ---- args ----
if [ "$#" -ne 2 ]; then
  echo "usage: $0 <conversations_dir> <split_name:  train|eval>" >&2
  exit 1
fi
SRC="$1"
SPLIT="$2"

if [ ! -d "$SRC" ]; then
  echo "error: conversations dir not found: $SRC" >&2
  exit 1
fi

# ---- output dir names per split ----
if [ "$SPLIT" = "train" ]; then
  TOK="tokens";       FEAT="feats";       PAIRS="pairs"
else
  TOK="tokens_${SPLIT}"; FEAT="feats_${SPLIT}"; PAIRS="pairs_${SPLIT}"
fi

mkdir -p "$TOK" "$FEAT" "$PAIRS"

echo "============================================================"
echo " split=$SPLIT"
echo " source : $SRC"
echo " tokens : $TOK/   feats: $FEAT/   pairs: $PAIRS/"
echo "============================================================"

# ---- Stage A: audio -> dual-channel Mimi tokens ----
echo
echo "[Stage A] CANDOR audio -> tokens ($TOK/)"
python scripts/candor_preprocess.py --root "$SRC" --out-dir "$TOK/"

# ---- Stage B: frozen Moshi -> cached features (per conversation) ----
echo
echo "[Stage B] frozen Moshi -> features ($FEAT/)"
shopt -s nullglob
for f in "$TOK"/*.npy; do
  id="$(basename "$f" .npy)"
  out="$FEAT/$id.npz"
  if [ -f "$out" ]; then
    echo "  [skip] $id (features exist)"
    continue
  fi
  echo "  === $id ==="
  python scripts/moshi_extract_features.py --tokens "$f" --out "$out"
done

# ---- pair via symlinks (no duplication) ----
echo
echo "[pair] linking matching .npy + .npz into $PAIRS/"
rm -f "$PAIRS"/*.npy "$PAIRS"/*.npz 2>/dev/null || true
for f in "$TOK"/*.npy; do
  id="$(basename "$f" .npy)"
  if [ -f "$FEAT/$id.npz" ]; then
    ln -sf "../$TOK/$id.npy"  "$PAIRS/$id.npy"
    ln -sf "../$FEAT/$id.npz" "$PAIRS/$id.npz"
  fi
done

# ---- summary ----
n_tok=$(ls "$TOK"/*.npy 2>/dev/null | wc -l)
n_feat=$(ls "$FEAT"/*.npz 2>/dev/null | wc -l)
n_pairs=$(ls "$PAIRS"/*.npz 2>/dev/null | wc -l)
echo
echo "[done] split=$SPLIT  tokens=$n_tok  features=$n_feat  pairs=$n_pairs"
if [ "$n_tok" -ne "$n_feat" ]; then
  echo "  WARNING: tokens != features — some conversations failed Stage B; re-run to retry."
fi
