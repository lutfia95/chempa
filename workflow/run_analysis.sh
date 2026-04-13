#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIGS=(
  "workflow/configs/primary_aliphatic_amines.longrun.toml"
  "workflow/configs/secondary_aliphatic_amines.longrun.toml"
  "workflow/configs/aromatic_amines.longrun.toml"
  "workflow/configs/others.longrun.toml"
)

echo "Starting long-run workflow series from: $ROOT_DIR"
echo "Results root: workflow/results_longrun"

for config in "${CONFIGS[@]}"; do
  echo
  echo "============================================================"
  echo "Running config: $config"
  echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  python -m workflow.main --config "$config"
  echo "Finished config: $config"
  echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo
echo "All four workflow runs completed."
echo "Inspect:"
echo "  workflow/results_longrun/primary_aliphatic_amines_longrun/"
echo "  workflow/results_longrun/secondary_aliphatic_amines_longrun/"
echo "  workflow/results_longrun/aromatic_amines_longrun/"
echo "  workflow/results_longrun/others_longrun/"
