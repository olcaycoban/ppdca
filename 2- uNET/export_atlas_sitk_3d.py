#!/usr/bin/env python3
"""
09_train25_test10_high_quality_registration.ipynb ile aynı kayıt + füzyon mantığından
3B FN|FP Plotly HTML üretir (U-Net sahnesine benzer).

Önkoşullar:
  pip install SimpleITK pandas numpy nrrd plotly scikit-image scipy

Örnek (tek hasta deneme):
  .venv/bin/python "2- uNET/export_atlas_sitk_3d.py" --pddca-root . --limit 1

Tüm test_offsite + üç yöntem (MV, STAPLE, deformable=top5’teki en iyi tek atlas):
  .venv/bin/python "2- uNET/export_atlas_sitk_3d.py" --pddca-root .

Not: Her hasta için 5 atlas × affine+deformable kayıt çok zaman alır; önce --limit ile deneyin.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nrrd
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_UNET = Path(__file__).resolve().parent
if str(_UNET) not in sys.path:
    sys.path.insert(0, str(_UNET))

try:
    import SimpleITK as sitk
except ImportError:
    print("SimpleITK gerekli: pip install SimpleITK", file=sys.stderr)
    sys.exit(1)

from pddca_plotly_3d import (  # noqa: E402
    STRUCTURE_NAMES,
    _HAS_PLOTLY_3D,
    write_pred_only_3d_html,
)


def load_patient_data(patient_id: str, data_dir: Path):
    patient_dir = data_dir / patient_id
    img_path = patient_dir / "img.nrrd"
    image_np, header = nrrd.read(str(img_path))
    image_sitk = sitk.GetImageFromArray(np.transpose(image_np, (2, 1, 0)))
    if header and "space directions" in header:
        try:
            sd = header["space directions"]
            spacing = [abs(sd[i][i]) for i in range(3)]
            image_sitk.SetSpacing(spacing)
        except Exception:
            pass
    structures = {}
    for name in STRUCTURE_NAMES:
        p = patient_dir / "structures" / f"{name}.nrrd"
        if p.exists():
            mask_np, _ = nrrd.read(str(p))
            mask_sitk = sitk.GetImageFromArray(np.transpose(mask_np, (2, 1, 0)))
            mask_sitk.CopyInformation(image_sitk)
            structures[name] = mask_sitk
    return image_sitk, structures


def dice_coefficient(pred, true):
    pa = sitk.GetArrayFromImage(pred).flatten()
    ta = sitk.GetArrayFromImage(true).flatten()
    inter = int(np.sum((pa > 0) & (ta > 0)))
    total = int(np.sum(pa > 0) + np.sum(ta > 0))
    if total == 0:
        return 1.0 if inter == 0 else 0.0
    return 2.0 * inter / total


def eval_dice_only(pred_structures, true_structures):
    results = {}
    for name in STRUCTURE_NAMES:
        if name not in pred_structures or name not in true_structures:
            continue
        results[name] = dice_coefficient(pred_structures[name], true_structures[name])
    return results


def register_affine(fixed_image, moving_image):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInterpolator(sitk.sitkLinear)
    init = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(init, inPlace=False)
    reg.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([4, 2, 1, 0.5])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    try:
        transform = reg.Execute(fixed_image, moving_image)
        return transform, sitk.Resample(
            moving_image,
            fixed_image,
            transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
    except Exception as e:
        print(f"Affine başarısız: {e}")
        return None, None


def register_deformable_bspline(fixed_image, moving_image_in_fixed_space):
    mesh_size = [2] * 3
    tx = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.02)
    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=5.0,
        numberOfIterations=50,
        convergenceMinimumValue=1e-3,
        convergenceWindowSize=5,
    )
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransformAsBSpline(tx, inPlace=True, scaleFactors=[1, 2])
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0.5])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    try:
        return reg.Execute(fixed_image, moving_image_in_fixed_space)
    except Exception as e:
        print(f"Deformable başarısız: {e}", flush=True)
        return None


def register_affine_and_deformable(fixed_image, moving_image):
    transform_affine, moving_affine = register_affine(fixed_image, moving_image)
    if transform_affine is None:
        return None, None
    transform_deformable = register_deformable_bspline(fixed_image, moving_affine)
    if transform_deformable is None:
        return transform_affine, moving_affine
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(transform_affine)
    composite.AddTransform(transform_deformable)
    try:
        return composite, sitk.Resample(
            moving_image,
            fixed_image,
            composite,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelID(),
        )
    except RuntimeError:
        return transform_affine, moving_affine


def single_atlas_affine_deformable(atlas_img, atlas_str, target_img):
    transform, _ = register_affine_and_deformable(target_img, atlas_img)
    if transform is None:
        return None
    pred = {}
    for name, mask in atlas_str.items():
        pred[name] = sitk.Resample(
            mask,
            target_img,
            transform,
            sitk.sitkNearestNeighbor,
            0.0,
            mask.GetPixelID(),
        )
    return pred


def majority_voting(masks_list):
    if len(masks_list) == 0:
        return None
    n = len(masks_list)
    total = np.zeros_like(sitk.GetArrayFromImage(masks_list[0]), dtype=np.int16)
    for m in masks_list:
        total += (sitk.GetArrayFromImage(m) > 0).astype(np.int16)
    fused = (total > n / 2).astype(np.uint8)
    out = sitk.GetImageFromArray(fused)
    out.CopyInformation(masks_list[0])
    return out


def staple_fusion(masks_list):
    if len(masks_list) == 0:
        return None
    cast_masks = [sitk.Cast(m, sitk.sitkUInt8) for m in masks_list]
    try:
        f = sitk.STAPLEImageFilter()
        f.SetMaximumIterations(30)
        f.SetForegroundValue(1)
        prob = f.Execute(cast_masks)
        return sitk.BinaryThreshold(prob, lowerThreshold=0.5)
    except Exception as e:
        print(f"STAPLE hatası: {e}", flush=True)
        return majority_voting(masks_list)


def sitk_structures_to_label_volume(structures: dict, ref_img: sitk.Image) -> np.ndarray:
    arr = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint8)
    for idx, name in enumerate(STRUCTURE_NAMES, start=1):
        if name not in structures:
            continue
        m = sitk.GetArrayFromImage(structures[name]) > 0
        arr[m] = idx
    return arr


def sitk_pred_dict_to_label_volume(fused: dict, ref_img: sitk.Image) -> np.ndarray:
    arr = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint8)
    for idx, name in enumerate(STRUCTURE_NAMES, start=1):
        if name not in fused:
            continue
        m = sitk.GetArrayFromImage(fused[name]) > 0
        arr[m] = idx
    return arr


def complete_patients_for_structures(pddca_root: Path, patient_ids: list[str]) -> list[str]:
    out = []
    for pid in patient_ids:
        sd = pddca_root / pid / "structures"
        if not sd.is_dir():
            continue
        if all((sd / f"{n}.nrrd").exists() for n in STRUCTURE_NAMES):
            out.append(pid)
    return out


def resolve_train_test_ids(split: dict, pddca_root: Path) -> tuple[list[str], list[str]]:
    """Notebook 09: train ilk 25 tam hasta; test için test_offsite (yoksa test)."""
    train_raw = split.get("train", [])
    test_raw = split.get("test_offsite") or split.get("test") or []
    train_ok = complete_patients_for_structures(pddca_root, train_raw)[:25]
    test_ok = complete_patients_for_structures(pddca_root, test_raw)
    if len(train_ok) < 25:
        allp = complete_patients_for_structures(
            pddca_root, train_raw + split.get("val", []) + split.get("test", [])
        )
        train_ok = allp[:25]
    if not test_ok and test_raw:
        test_ok = [p for p in test_raw if (pddca_root / p / "img.nrrd").is_file()]
    return train_ok, test_ok


def top5_atlas_ids_from_df_def(atlas_dir: Path) -> list[str]:
    pkl = atlas_dir / "train25_test10_hq_deformable_cache.pkl"
    if not pkl.is_file():
        return []
    df = pd.read_pickle(pkl)
    if not isinstance(df, pd.DataFrame) or "atlas_id" not in df.columns:
        return []
    g = df.groupby("atlas_id")["dice"].mean().sort_values(ascending=False)
    return [str(x) for x in g.head(5).index.tolist()]


def merge_manifest_scene(out_dir: Path, slug: str, patient_id: str, rel: str) -> None:
    mpath = out_dir / "manifest.json"
    if not mpath.is_file():
        return
    data = json.loads(mpath.read_text(encoding="utf-8"))
    for m in data.get("methods", []):
        if m.get("slug") == slug:
            m.setdefault("scenes", {})[patient_id] = rel
            break
    mpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Atlas (SimpleITK) → 3B FN|FP HTML")
    ap.add_argument("--pddca-root", type=Path, default=_REPO)
    ap.add_argument("--atlas-dir", type=Path, default=_REPO / "01-Atlas Based Methods")
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "site" / "public" / "pddca-viz-atlas",
        help="manifest + HTML çıktısı",
    )
    ap.add_argument("--limit", type=int, default=None, help="En fazla N test hastası")
    ap.add_argument("--embed-plotlyjs", action="store_true")
    ap.add_argument(
        "--methods",
        type=str,
        default="majority_voting,staple,deformable",
        help="Virgülle: majority_voting, staple, deformable",
    )
    args = ap.parse_args()

    if not _HAS_PLOTLY_3D:
        print("plotly ve scikit-image gerekli.", file=sys.stderr)
        sys.exit(1)

    split_path = args.pddca_root / "data_split.json"
    if not split_path.is_file():
        print("data_split.json yok:", split_path, file=sys.stderr)
        sys.exit(1)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids, test_ids = resolve_train_test_ids(split, args.pddca_root)
    if len(train_ids) < 1:
        print("Atlas için train hasta bulunamadı.", file=sys.stderr)
        sys.exit(1)
    if not test_ids:
        print("Test hasta listesi boş.", file=sys.stderr)
        sys.exit(1)

    top5 = top5_atlas_ids_from_df_def(args.atlas_dir)
    if len(top5) < 1:
        print("Uyarı: train25_test10_hq_deformable_cache.pkl yok veya boş; train[:5] kullanılıyor.", flush=True)
        top5 = [str(x) for x in train_ids[:5]]

    methods = {x.strip() for x in args.methods.split(",") if x.strip()}
    args.out.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for target_id in test_ids:
        if args.limit is not None and n_done >= args.limit:
            break
        tdir = args.pddca_root / target_id
        if not (tdir / "img.nrrd").is_file():
            print("Atlandı (img.nrrd yok):", target_id)
            continue
        try:
            target_img, target_str = load_patient_data(target_id, args.pddca_root)
        except Exception as e:
            print("Yüklenemedi", target_id, e, flush=True)
            continue
        if len(target_str) < len(STRUCTURE_NAMES):
            print("Eksik yapı:", target_id)
            continue

        gt_vol = sitk_structures_to_label_volume(target_str, target_img)

        if "majority_voting" in methods:
            preds_list = []
            for aid in top5:
                try:
                    a_img, a_str = load_patient_data(aid, args.pddca_root)
                except Exception:
                    continue
                pred = single_atlas_affine_deformable(a_img, a_str, target_img)
                if pred is not None:
                    preds_list.append(pred)
            if preds_list:
                fused = {}
                for name in STRUCTURE_NAMES:
                    masks = [p[name] for p in preds_list if name in p]
                    if masks:
                        fused[name] = majority_voting(masks)
                pred_vol = sitk_pred_dict_to_label_volume(fused, target_img)
                sub = args.out / "majority_voting" / target_id
                html = sub / "scene_3d.html"
                title = "Atlas — HQ deformable + çoğunluk oylaması (SimpleITK)"
                if write_pred_only_3d_html(
                    pred_vol,
                    html,
                    target_id,
                    method_label=title,
                    embed_plotlyjs=args.embed_plotlyjs,
                ):
                    rel = f"pddca-viz-atlas/majority_voting/{target_id}/scene_3d.html"
                    merge_manifest_scene(args.out, "majority_voting", target_id, rel)
                    print("  MV 3B:", target_id)

        if "staple" in methods:
            preds_list = []
            for aid in top5:
                try:
                    a_img, a_str = load_patient_data(aid, args.pddca_root)
                except Exception:
                    continue
                pred = single_atlas_affine_deformable(a_img, a_str, target_img)
                if pred is not None:
                    preds_list.append(pred)
            if preds_list:
                fused = {}
                for name in STRUCTURE_NAMES:
                    masks = [p[name] for p in preds_list if name in p]
                    if masks:
                        fused[name] = staple_fusion(masks)
                pred_vol = sitk_pred_dict_to_label_volume(fused, target_img)
                sub = args.out / "staple" / target_id
                html = sub / "scene_3d.html"
                title = "Atlas — HQ deformable + STAPLE (SimpleITK)"
                if write_pred_only_3d_html(
                    pred_vol,
                    html,
                    target_id,
                    method_label=title,
                    embed_plotlyjs=args.embed_plotlyjs,
                ):
                    rel = f"pddca-viz-atlas/staple/{target_id}/scene_3d.html"
                    merge_manifest_scene(args.out, "staple", target_id, rel)
                    print("  STAPLE 3B:", target_id)

        if "deformable" in methods:
            best_aid = top5[0]
            try:
                a_img, a_str = load_patient_data(best_aid, args.pddca_root)
            except Exception as e:
                print("  Deformable atlas yüklenemedi", best_aid, e, flush=True)
                a_img, a_str = None, None
            if a_img is not None:
                pred = single_atlas_affine_deformable(a_img, a_str, target_img)
                if pred:
                    pred_vol = sitk_pred_dict_to_label_volume(pred, target_img)
                    sub = args.out / "deformable" / target_id
                    html = sub / "scene_3d.html"
                    title = f"Atlas — HQ deformable (tek atlas {best_aid}, SimpleITK)"
                    if write_pred_only_3d_html(
                        pred_vol,
                        html,
                        target_id,
                        method_label=title,
                        embed_plotlyjs=args.embed_plotlyjs,
                    ):
                        rel = f"pddca-viz-atlas/deformable/{target_id}/scene_3d.html"
                        merge_manifest_scene(args.out, "deformable", target_id, rel)
                        print("  Deformable 3B:", target_id, "atlas", best_aid)

        n_done += 1
        print("Hasta işlendi:", target_id, flush=True)

    print("Tamam. manifest (varsa) güncellendi:", args.out / "manifest.json")


if __name__ == "__main__":
    main()
