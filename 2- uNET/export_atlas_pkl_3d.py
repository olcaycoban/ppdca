#!/usr/bin/env python3
"""
Atlas yöntemleri (varsayılan üç HQ pickle) için site çıktıları (U-Net / torch gerekmez).

Pickle’lar (09_train25_test10_high_quality_registration.ipynb):
  — Çoğu durumda pandas DataFrame: (target_id, atlas_id?, organ, dice) → **Dice özet JSON**
  — Nadiren dict + 3B numpy hacim → **FN|FP Plotly HTML** (plotly + scikit-image gerekir)

Varsayılan dosyalar (01-Atlas Based Methods/):
  train25_test10_hq_deformable_cache.pkl
  train25_test10_hq_majority_voting_deformable_cache.pkl
  train25_test10_hq_staple_deformable_cache.pkl

Çıktılar (site/public/pddca-viz-atlas/):
  atlas_dice_by_method.json  — her yöntem için test hastası × organ Dice
  manifest.json              — 3B sahne yolları (hacim pickle’ı yoksa scenes boş kalır)

Tanınmayan pickle için:  .venv/bin/python "2- uNET/export_atlas_pkl_3d.py" --inspect /yol/cache.pkl

Gereksinim: pandas numpy scipy nrrd; 3B için ek: plotly scikit-image
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

_REPO = Path(__file__).resolve().parent.parent
_UNET = Path(__file__).resolve().parent
if str(_UNET) not in sys.path:
    sys.path.insert(0, str(_UNET))

from pddca_plotly_3d import (  # noqa: E402
    STRUCTURE_NAMES,
    _HAS_PLOTLY_3D,
    load_pddca_patient,
    write_pred_only_3d_html,
)

_PID_KEY = re.compile(r"^[A-Za-z0-9\-_]{6,}$")
_PRED_KEYS = (
    "pred",
    "prediction",
    "pred_volume",
    "segmentation",
    "labels",
    "label",
    "y_pred",
    "atlas_label",
    "warped_labels",
    "fused",
)


def _unwrap_container(obj):
    if not isinstance(obj, dict):
        return obj
    for k in ("results", "cache", "data", "test", "test_predictions", "predictions", "per_patient"):
        if k in obj and isinstance(obj[k], (dict, list)):
            inner = obj[k]
            if isinstance(inner, dict) and inner:
                return inner
            if isinstance(inner, list) and inner:
                return inner
    return obj


def _array_from_value(v) -> np.ndarray | None:
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], np.ndarray):
        return np.asarray(v[0])
    return None


def extract_patient_pred_map(raw) -> dict[str, np.ndarray]:
    """pickle kökünden {hasta_id: uint8 3D etiket} çıkarır."""
    data = _unwrap_container(raw)
    out: dict[str, np.ndarray] = {}

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            pid = item.get("patient_id") or item.get("id") or item.get("case") or item.get("pid")
            if pid is None:
                continue
            pid = str(pid)
            arr = None
            for key in _PRED_KEYS:
                if key in item:
                    arr = _array_from_value(item[key])
                    if arr is not None:
                        break
            if arr is not None:
                out[pid] = np.asarray(arr)
        return out

    if isinstance(data, dict):
        for k, v in data.items():
            sk = str(k)
            if isinstance(v, np.ndarray):
                if _PID_KEY.match(sk):
                    out[sk] = v
                continue
            if not isinstance(v, dict):
                continue
            pid = sk if _PID_KEY.match(sk) else None
            if pid is None:
                pid = v.get("patient_id") or v.get("id") or v.get("case")
                if pid is not None:
                    pid = str(pid)
            if pid is None:
                continue
            arr = None
            for key in _PRED_KEYS:
                if key in v:
                    arr = _array_from_value(v[key])
                    if arr is not None:
                        break
            if arr is None and "volume" in v:
                arr = _array_from_value(v["volume"])
            if arr is not None:
                out[pid] = np.asarray(arr)
    return out


def _align_pred_shape(pred: np.ndarray, ref_shape: tuple[int, int, int]) -> np.ndarray | None:
    pred = np.asarray(pred)
    if pred.ndim != 3:
        return None
    if pred.shape == ref_shape:
        return pred.astype(np.uint8, copy=False)
    factors = tuple(ref_shape[i] / pred.shape[i] for i in range(3))
    if any(f <= 0 for f in factors):
        return None
    if all(abs(f - 1.0) < 0.02 for f in factors):
        sl = tuple(
            slice(0, min(pred.shape[i], ref_shape[i])) for i in range(3)
        )
        out = np.zeros(ref_shape, dtype=np.uint8)
        out[sl] = pred[sl].astype(np.uint8)
        return out
    try:
        z = ndi.zoom(pred.astype(np.float32), factors, order=0)
        z = np.round(z).astype(np.uint8)
        if z.shape != ref_shape:
            return None
        return z
    except Exception:
        return None


def inspect_pkl(path: Path) -> None:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print("===", path.name, "===")
    print("type:", type(obj))
    if pd is not None and isinstance(obj, pd.DataFrame):
        print("pandas DataFrame:", obj.shape, "columns:", list(obj.columns))
        print(obj.head(3))
    if isinstance(obj, dict):
        keys = list(obj.keys())[:40]
        print("dict keys (ilk 40):", keys)
        if keys:
            k0 = keys[0]
            print(f"  örnek [{k0!r}]:", type(obj[k0]))
            if isinstance(obj[k0], dict):
                print("    alt anahtarlar:", list(obj[k0].keys())[:30])
    if isinstance(obj, list):
        print("list len:", len(obj))
        if obj:
            print("elem[0] type:", type(obj[0]))
            if isinstance(obj[0], dict):
                print("elem[0] keys:", list(obj[0].keys()))
    m = extract_patient_pred_map(obj)
    print("extract_patient_pred_map →", len(m), "hasta")
    for pid, arr in list(m.items())[:3]:
        print(f"  {pid}: shape={arr.shape} dtype={arr.dtype} maxlabel={int(arr.max())}")


def _dice_dict_from_dataframe(df, test_ids: list[str]) -> dict[str, dict[str, float]]:
    """test_offsite hastaları için {patient_id: {organ: dice}}."""
    if pd is None or not isinstance(df, pd.DataFrame):
        return {}
    need = {"target_id", "organ", "dice"}
    if not need.issubset(set(df.columns)):
        return {}
    tset = set(test_ids)
    sub = df[df["target_id"].astype(str).isin(tset)].copy()
    if sub.empty:
        return {}
    g = sub.groupby(["target_id", "organ"], as_index=False)["dice"].mean()
    out: dict[str, dict[str, float]] = {}
    for _, row in g.iterrows():
        pid = str(row["target_id"])
        org = str(row["organ"])
        out.setdefault(pid, {})[org] = float(row["dice"])
    return out


def export_atlas_dice_json(
    methods_spec: list[tuple[str, str, str]],
    atlas_dir: Path,
    out_dir: Path,
    test_ids: list[str],
) -> Path:
    """Notebook cache DataFrame’lerinden atlas_dice_by_method.json yazar."""
    out_path = out_dir / "atlas_dice_by_method.json"
    payload = {
        "structureNames": STRUCTURE_NAMES,
        "testPatientIds": list(test_ids),
        "note": "09 notebook: deformable = 25 atlas çiftleri (aşağıda atlas seçerek tek çift görebilirsiniz); MV/STAPLE = en iyi 5 atlas ile birleşik tahmin Dice.",
        "methods": [],
    }
    if pd is None:
        print("Uyarı: pandas yok; atlas_dice_by_method.json yazılamadı.", file=sys.stderr)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path

    for slug, title, pkl_name in methods_spec:
        pkl_path = atlas_dir / pkl_name
        mentry = {
            "id": slug,
            "label": title,
            "slug": slug,
            "pkl": pkl_name,
            "diceKind": "mean_over_atlases" if slug == "deformable" else "fused_segmentation",
            "diceByPatient": {},
        }
        if pkl_path.is_file():
            raw = pickle.load(open(pkl_path, "rb"))
            if isinstance(raw, pd.DataFrame):
                mentry["diceByPatient"] = _dice_dict_from_dataframe(raw, test_ids)
            else:
                print("  Dice export: DataFrame değil, atlandı:", pkl_path.name)
        payload["methods"].append(mentry)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("atlas Dice JSON:", out_path)
    return out_path


def export_deformable_pairwise_and_meta(
    atlas_dir: Path,
    out_dir: Path,
    test_ids: list[str],
) -> None:
    """
    09_train25_test10_high_quality_registration.ipynb ile uyumlu:
    - Deformable cache: (target_id, atlas_id, organ, dice) → JSON (hedef × atlas × organ)
    - En iyi 5 atlas: df_def.groupby('atlas_id')['dice'].mean().nlargest(5) — MV/STAPLE ile aynı ölçüt
    """
    if pd is None:
        return
    pkl_path = atlas_dir / "train25_test10_hq_deformable_cache.pkl"
    if not pkl_path.is_file():
        return
    df = pd.read_pickle(pkl_path)
    if not isinstance(df, pd.DataFrame) or "atlas_id" not in df.columns:
        return
    tset = {str(x) for x in test_ids}
    sub = df[df["target_id"].astype(str).isin(tset)].copy()
    if sub.empty:
        return

    atlas_means = sub.groupby("atlas_id")["dice"].mean().sort_values(ascending=False)
    top5 = [str(x) for x in atlas_means.head(5).index.tolist()]
    all_atlas = sorted({str(x) for x in sub["atlas_id"].astype(str).unique()})
    ordered: list[str] = []
    for a in top5:
        if a in all_atlas and a not in ordered:
            ordered.append(a)
    for a in all_atlas:
        if a not in ordered:
            ordered.append(a)

    by_target: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in sub.iterrows():
        pid = str(row["target_id"])
        aid = str(row["atlas_id"])
        org = str(row["organ"])
        by_target.setdefault(pid, {}).setdefault(aid, {})[org] = float(row["dice"])

    out_dir.mkdir(parents=True, exist_ok=True)
    pairwise_path = out_dir / "atlas_deformable_pairwise.json"
    with open(pairwise_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sourcePkl": "train25_test10_hq_deformable_cache.pkl",
                "notebookRef": "09_train25_test10_high_quality_registration.ipynb",
                "top5AtlasIdsByMeanDice": top5,
                "atlasIdsDisplayOrder": ordered,
                "byTarget": by_target,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("atlas deformable pairwise:", pairwise_path)

    meta = {
        "notebookRef": "09_train25_test10_high_quality_registration.ipynb",
        "deformablePairwise": "25 atlas × test hedefi; her satır bir (hedef, atlas, organ) Dice",
        "trainAtlasIdsInCache": ordered,
        "trainAtlasCount": len(all_atlas),
        "top5AtlasIdsForFusion": top5,
        "fusionSettings": {
            "majority_voting": {
                "nAtlas": 5,
                "atlasSelection": "df_def.groupby('atlas_id')['dice'].mean().sort_values(ascending=False).head(5)",
            },
            "staple": {
                "nAtlas": 5,
                "atlasSelection": "MV ile aynı (notebook 7.4)",
            },
            "weighted_voting_ncc": {
                "nAtlas": 5,
                "cachePkl": "train25_test10_hq_weighted_voting_deformable_cache.pkl",
                "note": "Site şu an MV + STAPLE + deformable çiftleri; WV ayrı eklenebilir",
            },
        },
    }
    meta_path = out_dir / "atlas_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("atlas meta:", meta_path)


DEFAULT_METHODS: list[tuple[str, str, str]] = [
    ("deformable", "Atlas — HQ deformable", "train25_test10_hq_deformable_cache.pkl"),
    (
        "majority_voting",
        "Atlas — HQ deformable + çoğunluk oylaması",
        "train25_test10_hq_majority_voting_deformable_cache.pkl",
    ),
    ("staple", "Atlas — HQ deformable + STAPLE", "train25_test10_hq_staple_deformable_cache.pkl"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Atlas pickle → FN|FP scene_3d.html + manifest")
    ap.add_argument("--pddca-root", type=Path, default=_REPO, help="data_split.json ve hasta klasörleri")
    ap.add_argument(
        "--atlas-dir",
        type=Path,
        default=_REPO / "01-Atlas Based Methods",
        help="pickle dosyalarının klasörü",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "site" / "public" / "pddca-viz-atlas",
        help="Çıktı kökü",
    )
    ap.add_argument("--embed-plotlyjs", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="En fazla N hasta işle")
    ap.add_argument("--inspect", type=Path, default=None, help="Pickle yapısını yazdır ve çık")
    args = ap.parse_args()

    if args.inspect:
        if not args.inspect.is_file():
            print("Dosya yok:", args.inspect, file=sys.stderr)
            sys.exit(1)
        inspect_pkl(args.inspect)
        return

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

    args.out.mkdir(parents=True, exist_ok=True)

    export_atlas_dice_json(DEFAULT_METHODS, args.atlas_dir, args.out, test_ids)
    export_deformable_pairwise_and_meta(args.atlas_dir, args.out, test_ids)

    manifest = {
        "layout": "fnfp",
        "structureNames": STRUCTURE_NAMES,
        "methods": [],
    }

    if not _HAS_PLOTLY_3D:
        print(
            "Uyarı: plotly/scikit-image yok; 3B HTML atlandı (Dice JSON yazıldı). pip install plotly scikit-image",
            file=sys.stderr,
        )
        for slug, title, pkl_name in DEFAULT_METHODS:
            manifest["methods"].append(
                {"id": slug, "label": title, "slug": slug, "pkl": pkl_name, "scenes": {}}
            )
        with open(args.out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print("manifest:", args.out / "manifest.json")
        return

    for slug, title, pkl_name in DEFAULT_METHODS:
        pkl_path = args.atlas_dir / pkl_name
        entry = {"id": slug, "label": title, "slug": slug, "pkl": pkl_name, "scenes": {}}
        if not pkl_path.is_file():
            print("Uyarı: pickle yok, atlanıyor:", pkl_path)
            manifest["methods"].append(entry)
            continue

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        if pd is not None and isinstance(raw, pd.DataFrame):
            print(
                "  3B atlandı (pickle Dice tablosu):",
                pkl_path.name,
                "— birleşik etiket hacmini .npy/.npz veya ayrı pickle ile verirseniz 3B üretilebilir.",
            )
            manifest["methods"].append(entry)
            continue
        pred_map = extract_patient_pred_map(raw)
        if not pred_map:
            print("Uyarı: 3B tahmin çıkarılamadı:", pkl_path.name, file=sys.stderr)
            print("  İpucu: --inspect", pkl_path, file=sys.stderr)
            manifest["methods"].append(entry)
            continue

        n_done = 0
        for pid in test_ids:
            if args.limit is not None and n_done >= args.limit:
                break
            if pid not in pred_map:
                continue
            pdir = args.pddca_root / pid
            if not (pdir / "img.nrrd").is_file():
                print("  img.nrrd yok, atlandı:", pid)
                continue
            _, gt = load_pddca_patient(pid, args.pddca_root)
            pred = _align_pred_shape(pred_map[pid], gt.shape)
            if pred is None:
                print(f"  Şekil uyuşmazlığı {pid}: pred {pred_map[pid].shape} vs gt {gt.shape}")
                continue
            pred = np.clip(pred, 0, 9).astype(np.uint8)
            sub = args.out / slug / pid
            html_path = sub / "scene_3d.html"
            if write_pred_only_3d_html(
                pred,
                html_path,
                pid,
                method_label=title,
                embed_plotlyjs=args.embed_plotlyjs,
            ):
                entry["scenes"][pid] = f"pddca-viz-atlas/{slug}/{pid}/scene_3d.html"
                print("  OK", slug, pid)
                n_done += 1

        manifest["methods"].append(entry)

    with open(args.out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("manifest:", args.out / "manifest.json")


if __name__ == "__main__":
    main()
