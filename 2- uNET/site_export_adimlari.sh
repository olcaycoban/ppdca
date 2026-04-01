#!/usr/bin/env bash
# PDDCA test_offsite → site/public/pddca-viz (PNG + scene_3d.html + manifest.json)
# Model: best_unet_v2.pth (2- uNET/ içinde olmalı)

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "=== Adım 1: Sanal ortam ==="
test -x .venv/bin/python || { echo "Hata: .venv yok → $REPO/.venv"; exit 1; }

echo "=== Adım 2: Model ==="
MODEL="2- uNET/best_unet_v2.pth"
test -f "$MODEL" || { echo "Hata: $MODEL yok"; exit 1; }

echo "=== Adım 3a: TEK hasta denemesi (--limit 1) ==="
.venv/bin/python "2- uNET/export_pddca_site_assets.py" --model "$MODEL" --limit 1

echo ""
echo "=== Adım 3b: Tam 10 hasta (şimdi çalıştırmak için) ==="
echo "  cd \"$REPO\" && .venv/bin/python \"2- uNET/export_pddca_site_assets.py\" --model \"$MODEL\""
echo ""
echo "=== Adım 4: Site ==="
echo "  cd \"$REPO/site\" && npm start"
echo "  Tarayıcı: http://localhost:3000  (Cmd+Shift+R)"
