#!/usr/bin/env python3
"""
Mevcut site/public/pddca-viz/manifest.json içindeki her hasta için dicePercent ve organMetrics
hesaplar (tam hacim; img.nrrd + model gerekir). PNG ve scene_3d.html dosyalarına dokunmaz.

Sadece mevcut scene_3d.html hover sayılarından organMetrics doldurmak için (model gerekmez):
  python3 "2- uNET/fill_manifest_organ_metrics_from_scene3d.py"

Kullanım (repo kökü):
  .venv/bin/python "2- uNET/fill_manifest_dice.py" --model "2- uNET/best_unet_v2.pth"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
UNET_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="manifest.json dicePercent doldur")
    ap.add_argument("--model", type=Path, required=True, help=".pth checkpoint")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "site/public/pddca-viz/manifest.json",
    )
    ap.add_argument("--pddca-root", type=Path, default=REPO)
    args = ap.parse_args()

    mp = args.manifest
    if not mp.is_file():
        print("manifest yok:", mp, file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_exp", UNET_DIR / "export_pddca_site_assets.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)

    with open(mp, encoding="utf-8") as f:
        m = json.load(f)

    mod = args.model.expanduser()
    if not mod.is_file():
        mod = (Path.cwd() / mod).resolve()
    if not mod.is_file():
        print("model yok:", args.model, file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = e.UNet(in_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load(mod, map_location=device))
    model.eval()
    print("Model:", mod, "| device:", device)

    for p in m.get("patients", []):
        pid = p["id"]
        pdir = args.pddca_root / pid
        if not (pdir / "img.nrrd").is_file():
            print("Atlandı (veri yok):", pid)
            continue
        print("Dice:", pid)
        img, gt = e.load_pddca_patient(pid, args.pddca_root)
        pred = e.run_inference(img, model, device)
        om = e.compute_organ_metrics(gt, pred)
        p["organMetrics"] = om
        p["dicePercent"] = {k: v["dicePercent"] for k, v in om.items()}

    with open(mp, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print("Tamam:", mp)


if __name__ == "__main__":
    main()
