#!/usr/bin/env bash
# Create base⊕v3.1 model soups for an alpha sweep. CPU-only. Run inside the
# serve image on the host. alpha = weight on v3.1 (0=base, 1=v3.1).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE:-/data_nvme/grj-projects/models/Qwen3-14B}"
V31="${V31:-/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage14b-1x8-hf-ckpt199}"
OUT_ROOT="${OUT_ROOT:-/data_nvme/grj-projects/models}"
ALPHAS="${ALPHAS:-0.25 0.50 0.75}"

for a in $ALPHAS; do
  tag="${a#*.}"            # 0.25 -> 25, 0.50 -> 50, 0.75 -> 75
  out="$OUT_ROOT/Qwen3-14B-soup${tag}"
  if [ -f "$out/model.safetensors.index.json" ] && ls "$out"/*.safetensors >/dev/null 2>&1; then
    echo "=== soup${tag} already exists, skip ==="
    continue
  fi
  echo "=== soup alpha=$a -> $out ==="
  python3 "$HERE/make_soup.py" --base "$BASE" --ft "$V31" --alpha "$a" --out "$out"
done
echo ALL_SOUPS_DONE
