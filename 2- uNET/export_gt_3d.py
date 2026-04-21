#!/usr/bin/env python3
"""
10 test_offsite hastası için Ground Truth 3B sahne üretir (model gerekmez).

Çıktı: site/public/pddca-viz/gt/<patient_id>/scene_3d.html
Manifest: site/public/pddca-viz/manifest.json → her hastaya "gt3d" alanı eklenir

Kullanım:
  .venv/bin/python "2- uNET/export_gt_3d.py" --pddca-root .
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_UNET = Path(__file__).resolve().parent
if str(_UNET) not in sys.path:
    sys.path.insert(0, str(_UNET))

from pddca_plotly_3d import (  # noqa: E402
    _HAS_PLOTLY_3D,
    load_pddca_patient,
    write_gt_only_3d_html,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="GT-only 3B sahne üretici")
    ap.add_argument("--pddca-root", type=Path, default=_REPO)
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "site" / "public" / "pddca-viz",
        help="Çıktı kökü (manifest.json ile aynı klasör)",
    )
    ap.add_argument("--embed-plotlyjs", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not _HAS_PLOTLY_3D:
        print("plotly + scikit-image gerekli: pip install plotly scikit-image", file=sys.stderr)
        sys.exit(1)

    split_path = args.pddca_root / "data_split.json"
    if not split_path.is_file():
        print("data_split.json bulunamadı:", split_path, file=sys.stderr)
        sys.exit(1)

    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    test_ids = split.get("test_offsite") or split.get("test") or []
    if not test_ids:
        print("test_offsite boş.", file=sys.stderr)
        sys.exit(1)

    manifest_path = args.out / "manifest.json"
    if not manifest_path.is_file():
        print("manifest.json bulunamadı:", manifest_path, file=sys.stderr)
        sys.exit(1)

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
        try:
            _, gt_vol = load_pddca_patient(pid, args.pddca_root)
        except Exception as e:
            print("Yüklenemedi", pid, e, flush=True)
            continue

        out_html = args.out / "gt" / pid / "scene_3d.html"
        if write_gt_only_3d_html(
            gt_vol,
            out_html,
            pid,
            embed_plotlyjs=args.embed_plotlyjs,
        ):
            rel = f"pddca-viz/gt/{pid}/scene_3d.html"
            if pid in patient_map:
                patient_map[pid]["gt3d"] = rel
            print("  GT 3B:", pid)
            n_done += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("manifest güncellendi:", manifest_path)


if __name__ == "__main__":
    main()
