#!/usr/bin/env python3
"""
manifest.json içine organMetrics yazar: scene_3d.html'deki Plotly hovertemplate
metinlerinden GT / tahmin / FN / FP vokselleri okunur. Model veya img.nrrd gerekmez.

Kullanım (repo kökü):
  python3 "2- uNET/fill_manifest_organ_metrics_from_scene3d.py"
  python3 "2- uNET/fill_manifest_organ_metrics_from_scene3d.py" --manifest site/public/pddca-viz/manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

HOVER_ROW = re.compile(
    r"<b>(?P<title>[^<]+)</b><br><b>(?P<organ>[^<]+)</b><br>Vokseller:\s*(?P<n>[\d,]+)",
    re.DOTALL,
)

TITLE_GT = "Ground truth (3D)"
TITLE_PRED = "Model tahmini (3D)"
TITLE_FN = "GT ≠ tahmin (GT etiketi)"
TITLE_FP = "Tahmin ≠ GT (tahmin etiketi)"


def decode_plotly_hover(raw: str) -> str:
    """JSON string içindeki \\uXXXX kaçışlarını çöz."""
    return raw.encode("utf-8").decode("unicode_escape")


def parse_scene3d_html(path: Path) -> dict[str, dict[str, int]]:
    """Organ -> {gt?, pred?, fn, fp} (panelde iz yoksa gt/pred anahtarı hiç olmayabilir)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    raw_templates = re.findall(r'"hovertemplate":"([^"]*)"', text)
    buckets: dict[str, dict[str, int]] = {}
    title_key = {
        TITLE_GT: "gt",
        TITLE_PRED: "pred",
        TITLE_FN: "fn",
        TITLE_FP: "fp",
    }
    for raw in raw_templates:
        dec = decode_plotly_hover(raw)
        m = HOVER_ROW.search(dec)
        if not m:
            continue
        title = m.group("title").strip()
        organ = m.group("organ").strip()
        n = int(m.group("n").replace(",", ""))
        key = title_key.get(title)
        if key is None:
            continue
        if organ not in buckets:
            buckets[organ] = {}
        buckets[organ][key] = n
    return buckets


def buckets_to_organ_metrics(
    structure_names: list[str],
    buckets: dict[str, dict[str, int]],
    dice_fallback: dict[str, float | None],
) -> dict[str, dict]:
    """export_pddca_site_assets.compute_organ_metrics ile aynı şema."""
    out: dict[str, dict] = {}
    for organ in structure_names:
        b = buckets.get(organ, {})
        fn = int(b.get("fn", 0))
        fp = int(b.get("fp", 0))
        gt_raw = b.get("gt")
        pred_raw = b.get("pred")

        if gt_raw is None and pred_raw is None:
            d0 = dice_fallback.get(organ)
            if fn == 0 and fp == 0:
                out[organ] = {
                    "dicePercent": d0,
                    "gtVoxels": 0,
                    "predVoxels": 0,
                    "tpVoxels": 0,
                    "fnVoxels": 0,
                    "fpVoxels": 0,
                }
                continue
            print(
                f"  Uyarı {organ}: GT/tahmin hover yok; FN={fn} FP={fp}. Sadece manifest DSC korunuyor.",
                file=sys.stderr,
            )
            out[organ] = {
                "dicePercent": d0,
                "gtVoxels": 0,
                "predVoxels": 0,
                "tpVoxels": 0,
                "fnVoxels": fn,
                "fpVoxels": fp,
            }
            continue

        if gt_raw is None and pred_raw is not None:
            pred = int(pred_raw)
            tp = pred - fp
            gt = tp + fn
        elif pred_raw is None and gt_raw is not None:
            gt = int(gt_raw)
            tp = gt - fn
            pred = tp + fp
        else:
            gt = int(gt_raw or 0)
            pred = int(pred_raw or 0)

        tp_via_gt = gt - fn
        tp_via_pred = pred - fp
        if tp_via_gt != tp_via_pred:
            tp = max(0, (tp_via_gt + tp_via_pred) // 2)
            print(
                f"  Uyarı {organ}: TP ortalaması kullanıldı (GT−FN={tp_via_gt}, Tahmin−FP={tp_via_pred}).",
                file=sys.stderr,
            )
        else:
            tp = tp_via_gt

        if gt == 0 and pred == 0:
            d0 = dice_fallback.get(organ)
            out[organ] = {
                "dicePercent": d0,
                "gtVoxels": 0,
                "predVoxels": 0,
                "tpVoxels": 0,
                "fnVoxels": fn,
                "fpVoxels": fp,
            }
            continue

        dsc = 200.0 * tp / (gt + pred) if (gt + pred) > 0 else None
        out[organ] = {
            "dicePercent": None if dsc is None else round(float(dsc), 1),
            "gtVoxels": gt,
            "predVoxels": pred,
            "tpVoxels": tp,
            "fnVoxels": fn,
            "fpVoxels": fp,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="manifest.json organMetrics: scene_3d.html hover'dan oku"
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "site/public/pddca-viz/manifest.json",
    )
    ap.add_argument(
        "--public-root",
        type=Path,
        default=REPO / "site/public",
        help="scene3d yollarının kökü (pddca-viz/...)",
    )
    args = ap.parse_args()
    mp = args.manifest
    if not mp.is_file():
        print("manifest yok:", mp, file=sys.stderr)
        sys.exit(1)

    with open(mp, encoding="utf-8") as f:
        m = json.load(f)

    for p in m.get("patients", []):
        rel = p.get("scene3d")
        if not rel:
            print("Atlandı (scene3d yok):", p.get("id"), file=sys.stderr)
            continue
        html_path = args.public_root / rel
        if not html_path.is_file():
            print("HTML yok:", html_path, file=sys.stderr)
            continue
        pid = p["id"]
        print("organMetrics ←", pid)
        buckets = parse_scene3d_html(html_path)
        names = m.get("structureNames") or []
        om = buckets_to_organ_metrics(names, buckets, p.get("dicePercent") or {})
        p["organMetrics"] = om
        # manifest'teki dicePercent ile karşılaştır (3B mesh ile küçük fark olabilir)
        old_dice = p.get("dicePercent") or {}
        for name, row in om.items():
            new_d = row.get("dicePercent")
            old_d = old_dice.get(name)
            if new_d is not None and old_d is not None and abs(new_d - old_d) > 0.6:
                print(
                    f"  DSC fark {pid}/{name}: manifest={old_d} scene3d={new_d}",
                    file=sys.stderr,
                )

    with open(mp, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print("Tamam:", mp)


if __name__ == "__main__":
    main()
