#!/usr/bin/env bash
# Download Qwen/Qwen3.6-27B (51.8GB, 15 safetensors) from ModelScope into
# /home/h800/grj-projects/models/Qwen3.6-27B on the .191 training host.
#
# Idempotent: modelscope's snapshot_download skips files already present
# with matching size. Safe to re-run after a network blip.
#
# Runs in a python:3.10-slim container so we don't pollute the host. The
# download cache lands at /home/h800/grj-projects/models/.cache/modelscope
# and the final layout is symlinked to .../models/Qwen3.6-27B.
#
# License: apache-2.0 (per ModelScope metadata). 15 files × ~3.9GB each
# plus tokenizer/config; 51.8 GB total.
#
# Run on .191 (or invoke from DEV via:
#   ssh -i ~/.ssh/id_ed25519 h800@172.16.120.191 'bash -s' \
#     < scripts/deploy/download_qwen36_27b_191.sh
# )

set -euo pipefail

MODEL_ID="Qwen/Qwen3.6-27B"
ROOT="/home/h800/grj-projects/models"
CACHE_DIR="$ROOT/.cache/modelscope"
TARGET_LINK="$ROOT/Qwen3.6-27B"

mkdir -p "$CACHE_DIR"

echo "=========================================="
echo "Download $MODEL_ID -> $ROOT"
echo "  cache:  $CACHE_DIR"
echo "  link:   $TARGET_LINK"
echo "=========================================="

# Run modelscope SDK in an ephemeral python:3.10-slim container.
# Mount only the models tree so the container has no other access.
docker run --rm \
  --network host \
  -v "$ROOT":/models \
  -e MODELSCOPE_CACHE=/models/.cache/modelscope \
  python:3.10-slim bash -c '
    set -euo pipefail
    pip install --quiet --no-cache-dir "modelscope>=1.18.0" requests
    python - <<PYEOF
from modelscope.hub.snapshot_download import snapshot_download
path = snapshot_download(
    "Qwen/Qwen3.6-27B",
    cache_dir="/models/.cache/modelscope",
    revision="master",
)
print("DOWNLOADED_AT:", path)
PYEOF
  '

# Resolve final path inside the cache and symlink it for stable use.
# ModelScope's snapshot_download rewrites '.' -> '___' in directory names,
# so Qwen/Qwen3.6-27B lands at <cache>/Qwen/Qwen3___6-27B.
RESOLVED="$CACHE_DIR/Qwen/Qwen3___6-27B"
if [[ ! -d "$RESOLVED" ]]; then
  echo "ERROR: expected directory not found at $RESOLVED" >&2
  echo "Inspecting cache layout:" >&2
  ls -la "$CACHE_DIR" >&2 || true
  ls -la "$CACHE_DIR/Qwen" 2>/dev/null >&2 || true
  exit 1
fi

ln -sfn "$RESOLVED" "$TARGET_LINK"

echo
echo "=== verification ==="
ls -la "$TARGET_LINK"/ | head
echo "--- size ---"
du -sh "$TARGET_LINK/"
echo "--- safetensors count ---"
ls "$TARGET_LINK/" | grep -c "^model-.*\.safetensors$" || true
echo "--- config snippets ---"
python3 -c "import json; d=json.load(open('$TARGET_LINK/config.json')); print({k: d[k] for k in ('model_type','architectures','hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','torch_dtype') if k in d})"
echo
echo "Download complete: $TARGET_LINK"
