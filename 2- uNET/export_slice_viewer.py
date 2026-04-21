#!/usr/bin/env python3
"""
10 test_offsite hastası için 2D axial kesit görüntüleyici HTML üretir.
Paneller: HU | Ground Truth | Tahmin (pred.nrrd varsa)

Kullanım:
  .venv/bin/python "2- uNET/export_slice_viewer.py" --pddca-root .
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_UNET = Path(__file__).resolve().parent
if str(_UNET) not in sys.path:
    sys.path.insert(0, str(_UNET))

from pddca_plotly_3d import load_pddca_patient   # noqa: E402
from pddca_slice_viewer import write_slice_viewer_html  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="2D kesit görüntüleyici üretici")
    ap.add_argument("--pddca-root", type=Path, default=_REPO)
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "site" / "public" / "pddca-viz",
    )
    ap.add_argument("--step", type=int, default=1,
                    help="Her kaç kesiti dahil et (1=hepsi)")
    ap.add_argument("--quality", type=int, default=72,
                    help="WebP kalitesi (50-85)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    split_path = args.pddca_root / "data_split.json"
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    test_ids = split.get("test_offsite") or split.get("test") or []

    manifest_path = args.out / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    patient_map = {p["id"]: p for p in manifest.get("patients", [])}

    n_done = 0
    for pid in test_ids:
        if args.limit is not None and n_done >= args.limit:
            break
        pdir = args.pddca_root / pid
        if not (pdir / "img.nrrd").is_file():
            print("img.nrrd yok, atlandı:", pid)
            continue

        print(f"  Yükleniyor: {pid} ...", end=" ", flush=True)
        try:
            img_vol, gt_vol = load_pddca_patient(pid, args.pddca_root)
        except Exception as e:
            print("HATA:", e)
            continue

        # Tahmin varsa yükle
        pred_vol = None
        pred_nrrd = args.out / pid / "pred.nrrd"
        if pred_nrrd.is_file():
            try:
                import nrrd
                pred_arr, _ = nrrd.read(str(pred_nrrd))
                pred_vol = pred_arr.astype(np.uint8)
                print("(pred OK)", end=" ", flush=True)
            except Exception as e:
                print(f"(pred yüklenemedi: {e})", end=" ", flush=True)

        organ_metrics = patient_map.get(pid, {}).get("organMetrics")
        out_html = args.out / pid / "slice_viewer.html"
        if write_slice_viewer_html(
            img_vol, gt_vol, out_html, pid,
            pred_vol=pred_vol,
            organ_metrics=organ_metrics,
            step=args.step,
            quality=args.quality,
        ):
            rel = f"pddca-viz/{pid}/slice_viewer.html"
            if pid in patient_map:
                patient_map[pid]["sliceViewer"] = rel
            size_mb = out_html.stat().st_size / 1e6
            print(f"OK ({size_mb:.1f} MB)")
            n_done += 1
        else:
            print("BAŞARISIZ")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n{n_done} hasta tamamlandı. manifest güncellendi.")


if __name__ == "__main__":
    main()
