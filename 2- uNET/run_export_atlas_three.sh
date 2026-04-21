#!/usr/bin/env bash
# Üç atlas pickle (deformable, majority voting, STAPLE) → site/public/pddca-viz-atlas/
# Önkoşul: 01-Atlas Based Methods/*.pkl ve repo kökünde data_split.json + test hastalarında img.nrrd
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
if [[ -x "${REPO}/.venv/bin/python" ]]; then
  PY="${REPO}/.venv/bin/python"
else
  PY="python3"
fi
exec "$PY" "${REPO}/2- uNET/export_atlas_pkl_3d.py" \
  --pddca-root "${REPO}" \
  --atlas-dir "${REPO}/01-Atlas Based Methods" \
  --out "${REPO}/site/public/pddca-viz-atlas" \
  "$@"

# 3B FN|FP (SimpleITK, uzun): ayrı çalıştırın — örn. tek hasta:
# "$PY" "${REPO}/2- uNET/export_atlas_sitk_3d.py" --pddca-root "${REPO}" --limit 1
