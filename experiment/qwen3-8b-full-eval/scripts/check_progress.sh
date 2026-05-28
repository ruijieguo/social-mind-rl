#!/usr/bin/env bash
# Quick progress snapshot — shows current progress of each eval task by parsing
# tqdm output from per-task logs.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$(cd "$HERE/.." && pwd)"

echo "=== Files in output/ ==="
ls -la "$EXP_ROOT/output/tombench" "$EXP_ROOT/output/hitom" 2>/dev/null

echo ""
echo "=== Latest progress per task log ==="
for f in "$EXP_ROOT"/logs/run_*_*.log; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .log)
  # Extract last tqdm line
  last_line=$(grep -aoE "[0-9]+%.*[0-9]+/[0-9]+ \[.*\]" "$f" | tail -1 || echo "no progress yet")
  echo "  $name: $last_line"
done

echo ""
echo "=== Final summary lines so far ==="
grep -h "^\[" "$EXP_ROOT"/logs/run_*_*.log 2>/dev/null | tail -30 || echo "(none yet)"

echo ""
echo "=== Background processes ==="
ps -ef | grep -E "(04_run_eval|parallel_eval)" | grep -v grep | awk '{print $2, $8, $9, $10, $11, $12, $13, $14}'

echo ""
echo "=== GPU utilization ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
