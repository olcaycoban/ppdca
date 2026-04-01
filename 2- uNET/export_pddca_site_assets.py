#!/usr/bin/env python3
"""
PDDCA test_offsite (10 hasta) için web sitesinde kullanılacak bileşik PNG'leri üretir:
  - gt.png   : CT + ground truth (Axial | Sagittal | Coronal)
  - pred.png : CT + model tahmini
  - fn.png   : CT + kaçırılan bölgeler (GT'de organ var, tahmin arka plan)
  - scene_3d.html : Plotly 2×2: GT | tahmin ; FN | FP (uyuşmayan vokseller, organ renkleriyle)

3D için: pip install plotly scikit-image

Notebook pddca_3d_visualization.ipynb ile aynı overlay, U-Net ve marching cubes stili.

Kullanım — proje sanal ortamındaki Python ile (sistem python3'te genelde torch yok):
  cd <repo_kökü>
  .venv/bin/python "2- uNET/export_pddca_site_assets.py" --model /yol/checkpoint.pth

Checkpoint yoksa: eğitim notebook'unu (Kaggle vb.) çalıştırıp üretilen .pth dosyasını kopyalayın veya
--model ile tam yol verin. İsteğe bağlı ortam değişkeni: PDDCA_MODEL (veya PDDCA_MODEL_PATH).

Görseller: ../site/public/pddca-viz/<hasta_id>/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nrrd
import scipy.ndimage as ndi

_REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    vpy = _REPO_ROOT / ".venv" / "bin" / "python"
    print(
        "Hata: 'torch' (PyTorch) bu Python ortamında yüklü değil.\n"
        f"Proje .venv kullanın, örneğin:\n  {vpy}\n"
        f'    "{Path(__file__)}" --model /yol/model.pth\n'
        "veya: cd repo_kökü && source .venv/bin/activate && python ...",
        file=sys.stderr,
    )
    sys.exit(1)

REPO_ROOT = _REPO_ROOT
PDDCA_DIR = REPO_ROOT
UNET_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = REPO_ROOT / "site" / "public" / "pddca-viz"

_SKIP_DIR_NAMES = frozenset({".venv", "node_modules", ".git", "build"})


def _discover_pth_under(roots: list[Path]) -> list[Path]:
    """Repo içinde .venv / node_modules içine inmeden *.pth ara."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
                for fn in filenames:
                    if not fn.endswith(".pth"):
                        continue
                    rp = (Path(dirpath) / fn).resolve()
                    if rp not in seen:
                        seen.add(rp)
                        found.append(rp)
        except OSError:
            continue
    return sorted(found)


def _pick_model_path() -> Path | None:
    """Öncelik: PDDCA_MODEL / PDDCA_MODEL_PATH → sabit adlar → repo genelinde tek .pth."""
    env = os.environ.get("PDDCA_MODEL") or os.environ.get("PDDCA_MODEL_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p

    for cand in [
        UNET_DIR / "best_unet_v2.pth",
        UNET_DIR / "hanseg_unet_best.pth",
        UNET_DIR / "pddca25_unet_best_recipe1.pth",
        UNET_DIR / "pddca25_unet_best_recipe2.pth",
        UNET_DIR / "pddca25_unet_best_recipe3.pth",
        UNET_DIR / "pddca25_unet_best_recipe2_full.pth",
        UNET_DIR / "pddca25_unet_best_recipe3_full.pth",
        UNET_DIR / "hanseg_unet_best_recipe1.pth",
    ]:
        if cand.is_file():
            return cand.resolve()

    # Tüm REPO_ROOT taraması binlerce NRRD ile çok yavaş; sadece tipik checkpoint klasörleri.
    search_roots = [
        UNET_DIR,
        REPO_ROOT / "models",
        REPO_ROOT / "checkpoints",
        REPO_ROOT / "outputs",
    ]
    all_pth = _discover_pth_under(search_roots)
    if len(all_pth) == 1:
        return all_pth[0]
    return None


def _print_model_help(discovered: list[Path]) -> None:
    print(
        "Bu dizinde veya repo kökünde henüz bir U-Net checkpoint (.pth) yok gibi görünüyor.\n",
        file=sys.stderr,
    )
    print("Ne yapmalı:", file=sys.stderr)
    print(
        "  1) Eğitimi notebook-5 / notebook-6 / notebook-7 ile çalıştırdıysanız, çıkan .pth dosyasını",
        file=sys.stderr,
    )
    print(
        f"     bu klasöre kopyalayın: {UNET_DIR}/",
        file=sys.stderr,
    )
    print(
        "     (ör. hanseg_unet_best.pth, pddca25_unet_best_recipe1.pth — notebook’taki CKPT_PATH adı).",
        file=sys.stderr,
    )
    print(
        "  2) Veya dosya neredeyse tam yolu verin:",
        file=sys.stderr,
    )
    print(
        f'     .venv/bin/python "{Path(__file__)}" --model /Users/siz/İndirilenler/model.pth',
        file=sys.stderr,
    )
    print(
        "  3) Ortam değişkeni: export PDDCA_MODEL=/yol/model.pth",
        file=sys.stderr,
    )
    if discovered:
        print("\nRepoda birden fazla .pth bulundu; hangisini kullanacağınızı seçin (--model):", file=sys.stderr)
        for p in discovered:
            print(f"  - {p}", file=sys.stderr)


STRUCTURE_NAMES = [
    "BrainStem",
    "Chiasm",
    "Mandible",
    "OpticNerve_L",
    "OpticNerve_R",
    "Parotid_L",
    "Parotid_R",
    "Submandibular_L",
    "Submandibular_R",
]
COLORS_3D = [
    (1, 0.2, 0.2, 0.7),
    (1, 0.8, 0, 0.7),
    (0.2, 0.8, 0.2, 0.7),
    (0, 0.6, 1, 0.7),
    (0.6, 0, 1, 0.7),
    (1, 0.5, 0, 0.7),
    (1, 0, 0.5, 0.7),
    (0, 0.8, 0.8, 0.7),
    (0.8, 0.6, 0.2, 0.7),
]

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from skimage.measure import marching_cubes

    _HAS_PLOTLY_3D = True
except ImportError:
    _HAS_PLOTLY_3D = False
    go = None  # type: ignore


def _rgb_plotly(col):
    return f"rgb({int(col[0] * 255)},{int(col[1] * 255)},{int(col[2] * 255)})"


def _organ_hover_html(panel: str | None, organ_name: str, n_vox: int) -> str:
    """Plotly mesh3d / scatter3d hover — panel başlığı + organ + vokseller."""
    nv = f"{int(n_vox):,}"
    if panel:
        return f"<b>{panel}</b><br><b>{organ_name}</b><br>Vokseller: {nv}<extra></extra>"
    return f"<b>{organ_name}</b><br>Vokseller: {nv}<extra></extra>"


def extract_surface_mesh(binary_vol, spacing=(1, 1, 1), step_size=2):
    if not _HAS_PLOTLY_3D:
        return None, None
    try:
        verts, faces, _, _ = marching_cubes(
            binary_vol.astype(np.float32), 0.5, spacing=spacing, step_size=step_size
        )
        return verts, faces
    except Exception:
        return None, None


def mesh3d_traces_for_volume(
    label_vol,
    step_size: int = 2,
    min_voxels: int = 10,
    *,
    hover_panel: str | None = None,
    binary_fallback_rgb: tuple[float, float, float] | None = None,
    binary_fallback_opacity: float = 0.82,
):
    """Organ başına mesh. Seyrek hata maskeleri için min_voxels düşük + step_size=1 önerilir."""
    if not _HAS_PLOTLY_3D:
        return []
    traces = []
    for c in range(1, 10):
        mask = (label_vol == c).astype(np.uint8)
        n_vox = int(mask.sum())
        if n_vox < min_voxels:
            continue
        verts, faces = extract_surface_mesh(mask, step_size=step_size)
        if verts is None:
            continue
        col = COLORS_3D[c - 1]
        oname = STRUCTURE_NAMES[c - 1]
        traces.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=_rgb_plotly(col[:3]),
                opacity=col[3],
                name=oname,
                flatshading=True,
                showlegend=False,
                hovertemplate=_organ_hover_html(hover_panel, oname, n_vox),
            )
        )

    if traces or binary_fallback_rgb is None:
        return traces

    n_raw_total = int(np.sum(label_vol > 0))
    union = (label_vol > 0).astype(np.uint8)
    if union.sum() < 1:
        return traces

    struct = ndi.generate_binary_structure(3, 1)
    try:
        union = ndi.binary_closing(union, structure=struct, iterations=1).astype(np.uint8)
        if union.sum() < 1:
            union = (label_vol > 0).astype(np.uint8)
    except Exception:
        pass
    try:
        union = ndi.binary_dilation(union, structure=struct, iterations=1).astype(np.uint8)
    except Exception:
        pass

    verts, faces = extract_surface_mesh(union, step_size=1)
    if verts is None:
        verts, faces = extract_surface_mesh((label_vol > 0).astype(np.uint8), step_size=1)
    if verts is None:
        base = (label_vol > 0).astype(np.uint8)
        for it in (2, 3, 4, 5):
            try:
                thick = ndi.binary_dilation(base, structure=struct, iterations=it).astype(np.uint8)
                verts, faces = extract_surface_mesh(thick, step_size=1)
                if verts is not None:
                    break
            except Exception:
                pass
    if verts is None:
        return traces
    r, g, b = binary_fallback_rgb
    bh = (
        f"<b>{hover_panel}</b><br>" if hover_panel else ""
    ) + f"<b>Birleşik yüzey</b><br>Vokseller (bu panelde): {n_raw_total:,}<extra></extra>"
    traces.append(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=_rgb_plotly((r, g, b)),
            opacity=binary_fallback_opacity,
            name="uyuşmazlık",
            flatshading=True,
            showlegend=False,
            hovertemplate=bh,
        )
    )
    return traces


def scatter3d_voxels_for_volume(
    label_vol,
    max_points: int = 28000,
    marker_size: float = 2.8,
    rng: np.random.Generator | None = None,
    hover_panel: str | None = None,
):
    """Marching cubes başarısız olduğunda uyuşmazlık voksellerini sınıf başına nokta bulutu + hover."""
    if not _HAS_PLOTLY_3D:
        return []
    if rng is None:
        rng = np.random.default_rng(0)
    total_pos = int(np.sum(label_vol > 0))
    if total_pos < 1:
        return []
    traces = []
    for c in range(1, 10):
        idx = np.argwhere(label_vol == c)
        n_full = len(idx)
        if n_full < 1:
            continue
        cap = max(1, int(max_points * n_full / total_pos))
        if n_full > cap:
            sel = rng.choice(n_full, size=cap, replace=False)
            idx = idx[sel]
        oname = STRUCTURE_NAMES[c - 1]
        col = COLORS_3D[c - 1]
        rgb = _rgb_plotly(col[:3])
        traces.append(
            go.Scatter3d(
                x=idx[:, 0],
                y=idx[:, 1],
                z=idx[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color=rgb, opacity=0.78, line=dict(width=0)),
                hovertemplate=_organ_hover_html(hover_panel, oname, n_full),
                showlegend=False,
            )
        )
    return traces


def traces_for_error_subplot(
    label_vol: np.ndarray,
    fallback_rgb: tuple[float, float, float],
    rng: np.random.Generator,
    hover_panel: str | None = None,
) -> list:
    """Önce mesh (+ kalınlaştırılmış birleşik yüzey), olmazsa voxel scatter."""
    t = mesh3d_traces_for_volume(
        label_vol,
        step_size=1,
        min_voxels=1,
        hover_panel=hover_panel,
        binary_fallback_rgb=fallback_rgb,
    )
    if t:
        return t
    return scatter3d_voxels_for_volume(label_vol, rng=rng, hover_panel=hover_panel)


def write_gt_pred_3d_html(
    gt_label_vol: np.ndarray,
    pred_label_vol: np.ndarray,
    out_html: Path,
    patient_id: str,
    step_size: int = 2,
    embed_plotlyjs: bool = False,
) -> bool:
    """2×2 Plotly: GT | tahmin ; FN (GT etiketiyle) | FP (tahmin etiketiyle)."""
    if not _HAS_PLOTLY_3D:
        return False
    # Uyuşmayan vokseller: alt satırda GT rengi / tahmin rengiyle yüzey
    disagree = gt_label_vol != pred_label_vol
    fn_vol = np.where((gt_label_vol > 0) & disagree, gt_label_vol, 0).astype(np.uint8)
    fp_vol = np.where((pred_label_vol > 0) & disagree, pred_label_vol, 0).astype(np.uint8)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "Ground truth (3D)",
            "Model tahmini (3D)",
            "GT’de olup tahminle uyuşmayan (FN / yanlış sınıf)",
            "Tahminde olup GT ile uyuşmayan (FP / yanlış sınıf)",
        ),
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
    )
    scene_kw = dict(
        xaxis=dict(title="X", backgroundcolor="white"),
        yaxis=dict(title="Y", backgroundcolor="white"),
        zaxis=dict(title="Z (kesit)", backgroundcolor="white"),
        bgcolor="#f8f9fa",
        aspectmode="data",
    )
    fig.update_layout(
        title_text=f"PDDCA — 3D organ yüzeyleri ({patient_id})",
        height=1120,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor="white",
        scene=scene_kw,
        scene2=dict(**scene_kw),
        scene3=dict(**scene_kw),
        scene4=dict(**scene_kw),
        showlegend=False,
    )
    for tr in mesh3d_traces_for_volume(
        gt_label_vol,
        step_size=step_size,
        min_voxels=10,
        hover_panel="Ground truth (3D)",
    ):
        fig.add_trace(tr, row=1, col=1)
    for tr in mesh3d_traces_for_volume(
        pred_label_vol,
        step_size=step_size,
        min_voxels=10,
        hover_panel="Model tahmini (3D)",
    ):
        fig.add_trace(tr, row=1, col=2)
    rng_fn = np.random.default_rng(zlib.adler32((patient_id + "|fn").encode()) & 0xFFFFFFFF)
    rng_fp = np.random.default_rng(zlib.adler32((patient_id + "|fp").encode()) & 0xFFFFFFFF)
    for tr in traces_for_error_subplot(
        fn_vol,
        (0.95, 0.25, 0.35),
        rng_fn,
        hover_panel="GT ≠ tahmin (GT etiketi)",
    ):
        fig.add_trace(tr, row=2, col=1)
    for tr in traces_for_error_subplot(
        fp_vol,
        (0.95, 0.55, 0.1),
        rng_fp,
        hover_panel="Tahmin ≠ GT (tahmin etiketi)",
    ):
        fig.add_trace(tr, row=2, col=2)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    include_js = True if embed_plotlyjs else "cdn"
    fig.write_html(
        str(out_html),
        include_plotlyjs=include_js,
        config=dict(scrollZoom=True, displaylogo=False),
        full_html=True,
    )
    return True


def load_pddca_patient(patient_id: str, pddca_dir: Path):
    pdir = pddca_dir / patient_id
    img_np, _ = nrrd.read(str(pdir / "img.nrrd"))
    label = np.zeros(img_np.shape, dtype=np.uint8)
    for idx, name in enumerate(STRUCTURE_NAMES, start=1):
        p = pdir / "structures" / f"{name}.nrrd"
        if p.exists():
            mask, _ = nrrd.read(str(p))
            label[mask > 0] = idx
    return img_np, label


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.final_conv(x)


def preprocess_volume(img_np, target_size=256, hu_min=-200, hu_max=300):
    img = np.clip(img_np.astype(np.float32), hu_min, hu_max)
    img = (img - hu_min) / (hu_max - hu_min)
    n_slices = img.shape[2]
    slices_img = []
    for s in range(n_slices):
        sl_img = img[:, :, s]
        if target_size != sl_img.shape[0]:
            zf = target_size / sl_img.shape[0]
            sl_img = ndi.zoom(sl_img, zf, order=1)
        slices_img.append(sl_img)
    return np.array(slices_img, dtype=np.float32)


def predict_volume(model, preprocessed_slices, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(preprocessed_slices)):
            img_t = torch.from_numpy(preprocessed_slices[i : i + 1]).unsqueeze(0).to(device)
            logits = model(img_t)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            preds.append(pred)
    return np.stack(preds, axis=0)


def clean_prediction(pred_vol, num_classes=10, min_voxels=5):
    from scipy.ndimage import (
        label as nd_label,
        binary_opening,
        binary_closing,
        binary_fill_holes,
        generate_binary_structure,
        median_filter,
    )

    SMALL_ORGANS = {2, 4, 5}
    struct_3d = generate_binary_structure(3, 2)
    cleaned = np.zeros_like(pred_vol)

    for c in range(1, num_classes):
        mask = (pred_vol == c).astype(np.uint8)
        if mask.sum() < min_voxels:
            continue

        if c in SMALL_ORGANS:
            labeled_array, num_features = nd_label(mask, structure=struct_3d)
            if num_features == 0:
                continue
            component_sizes = np.bincount(labeled_array.ravel())
            component_sizes[0] = 0
            largest = component_sizes.argmax()
            best_mask = labeled_array == largest
            if best_mask.sum() >= min_voxels:
                cleaned[best_mask] = c
        else:
            mask = median_filter(mask, size=3)
            mask = (mask > 0.5).astype(np.uint8)
            mask = binary_opening(mask, structure=struct_3d, iterations=1).astype(np.uint8)
            mask = binary_closing(mask, structure=struct_3d, iterations=2).astype(np.uint8)
            mask = binary_fill_holes(mask).astype(np.uint8)
            labeled_array, num_features = nd_label(mask, structure=struct_3d)
            if num_features == 0:
                continue
            component_sizes = np.bincount(labeled_array.ravel())
            component_sizes[0] = 0
            largest = component_sizes.argmax()
            best_mask = labeled_array == largest
            if best_mask.sum() >= min_voxels:
                cleaned[best_mask] = c

    return cleaned


def normalize_ct_slice(sl_img):
    x = np.clip(sl_img.astype(np.float32), -200, 300)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x


def overlay_labels(sl_img, sl_lbl):
    sl_img = normalize_ct_slice(sl_img)
    rgb = np.stack([sl_img] * 3, axis=-1)
    for c in range(1, 10):
        m = sl_lbl == c
        if m.any():
            rgb[m, 0] = COLORS_3D[c - 1][0]
            rgb[m, 1] = COLORS_3D[c - 1][1]
            rgb[m, 2] = COLORS_3D[c - 1][2]
    return np.clip(rgb, 0, 1)


def overlay_false_negative(sl_img, gt_lbl, pred_lbl):
    """GT'de etiketli, tahminde arka plan (0) kalan bölgeler — organ rengiyle."""
    sl_img = normalize_ct_slice(sl_img)
    rgb = np.stack([sl_img] * 3, axis=-1)
    miss = (gt_lbl > 0) & (pred_lbl == 0)
    for c in range(1, 10):
        m = miss & (gt_lbl == c)
        if m.any():
            rgb[m, 0] = COLORS_3D[c - 1][0]
            rgb[m, 1] = COLORS_3D[c - 1][1]
            rgb[m, 2] = COLORS_3D[c - 1][2]
    return np.clip(rgb, 0, 1)


def save_orthogonal_panel(out_path: Path, suptitle: str, rgb_views: tuple):
    """rgb_views: (axial_rgb, sagittal_rgb, coronal_rgb) each (H,W,3) float 0..1"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, im, title in zip(axes, rgb_views, ("Axial", "Sagittal", "Coronal")):
        ax.imshow(im, origin="lower")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def orthogonal_rgb_views(img_vol, gt_vol, pred_vol, mode: str):
    """mode: 'gt' | 'pred' | 'fn'"""
    mid_z = gt_vol.shape[2] // 2
    mid_h, mid_w = gt_vol.shape[0] // 2, gt_vol.shape[1] // 2

    if mode == "gt":

        def paint(i, g, p):
            return overlay_labels(i, g)

    elif mode == "pred":

        def paint(i, g, p):
            return overlay_labels(i, p)

    else:

        def paint(i, g, p):
            return overlay_false_negative(i, g, p)

    v0 = paint(img_vol[:, :, mid_z], gt_vol[:, :, mid_z], pred_vol[:, :, mid_z])
    v1 = paint(img_vol[mid_h, :, :].T, gt_vol[mid_h, :, :].T, pred_vol[mid_h, :, :].T)
    v2 = paint(img_vol[:, mid_w, :].T, gt_vol[:, mid_w, :].T, pred_vol[:, mid_w, :].T)
    return (v0, v1, v2)


def compute_organ_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, dict]:
    """Organ başına DSC (%) ve vokseller: GT, tahmin, TP, FN, FP (sınıf c için)."""
    out: dict[str, dict] = {}
    for c in range(1, 10):
        name = STRUCTURE_NAMES[c - 1]
        g = gt == c
        p = pred == c
        g_n = int(g.sum())
        p_n = int(p.sum())
        tp = int((g & p).sum())
        fn_n = g_n - tp
        fp_n = p_n - tp
        if g_n == 0 and p_n == 0:
            out[name] = {
                "dicePercent": None,
                "gtVoxels": 0,
                "predVoxels": 0,
                "tpVoxels": 0,
                "fnVoxels": 0,
                "fpVoxels": 0,
            }
            continue
        dsc = 200.0 * tp / (g_n + p_n)
        out[name] = {
            "dicePercent": round(float(dsc), 1),
            "gtVoxels": g_n,
            "predVoxels": p_n,
            "tpVoxels": tp,
            "fnVoxels": fn_n,
            "fpVoxels": fp_n,
        }
    return out


def dice_percent_per_structure(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    """Sadece yüzde sözlüğü (geriye dönük)."""
    return {k: v["dicePercent"] for k, v in compute_organ_metrics(gt, pred).items()}


def run_inference(img_vol, model, device):
    pre = preprocess_volume(img_vol)
    pred_256 = predict_volume(model, pre, device)
    oh, ow = img_vol.shape[0], img_vol.shape[1]
    if pred_256.shape[1] != oh or pred_256.shape[2] != ow:
        pred_resized = []
        for s in range(pred_256.shape[0]):
            sl = ndi.zoom(
                pred_256[s],
                (oh / pred_256.shape[1], ow / pred_256.shape[2]),
                order=0,
            )
            pred_resized.append(sl)
        pred_resized = np.stack(pred_resized, axis=0)
    else:
        pred_resized = pred_256
    pred_hwz = np.transpose(pred_resized, (1, 2, 0))
    return clean_prediction(pred_hwz, num_classes=10)


def main():
    ap = argparse.ArgumentParser(description="PDDCA site PNG export")
    ap.add_argument(
        "--pddca-root",
        type=Path,
        default=PDDCA_DIR,
        help="data_split.json ve hasta klasörlerinin kökü",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="U-Net state_dict (.pth). Yoksa UNET_DIR içinde otomatik aranır.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Çıktı kökü (site/public/pddca-viz)")
    ap.add_argument("--no-3d", action="store_true", help="scene_3d.html (Plotly) üretme")
    ap.add_argument(
        "--mesh-step",
        type=int,
        default=2,
        metavar="N",
        help="Marching cubes step_size (notebook varsayılanı 2; büyütürseniz daha hızlı/kaba)",
    )
    ap.add_argument(
        "--embed-plotlyjs",
        action="store_true",
        help="Plotly.js'i her HTML'e göm (internet/CDN gerekmez; dosya ~3MB)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="img.nrrd bulunan hastalardan sırayla en fazla N tanesini işle (hızlı deneme).",
    )
    args = ap.parse_args()

    split_path = args.pddca_root / "data_split.json"
    if not split_path.is_file():
        print("data_split.json bulunamadı:", split_path, file=sys.stderr)
        sys.exit(1)

    with open(split_path) as f:
        split = json.load(f)
    test_ids = split.get("test_offsite") or split.get("test") or []
    if not test_ids:
        print("test_offsite listesi boş.", file=sys.stderr)
        sys.exit(1)

    search_roots = [
        UNET_DIR,
        REPO_ROOT / "models",
        REPO_ROOT / "checkpoints",
        REPO_ROOT / "outputs",
    ]
    discovered = _discover_pth_under(search_roots)

    if args.model is not None:
        p = args.model.expanduser()
        if not p.is_file():
            p = (Path.cwd() / p).resolve()
        if not p.is_file():
            print(f"Hata: --model dosyası bulunamadı: {args.model}", file=sys.stderr)
            sys.exit(1)
        model_path = p
    else:
        model_path = _pick_model_path()
        if model_path is None and len(discovered) > 1:
            print(
                "Birden fazla .pth bulundu; hangisini kullanacağınızı --model ile belirtin:",
                file=sys.stderr,
            )
            for x in discovered:
                print(f"  {x}", file=sys.stderr)
            sys.exit(1)

    if model_path is None or not model_path.is_file():
        _print_model_help(discovered)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model:", model_path, "| device:", device)

    if not args.no_3d and not _HAS_PLOTLY_3D:
        print(
            "Uyarı: plotly veya scikit-image yüklü değil; 3D HTML atlanıyor.\n"
            "  pip install plotly scikit-image",
            file=sys.stderr,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "structureNames": STRUCTURE_NAMES,
        "colorsRgb": [[int(c[i] * 255) for i in range(3)] for c in COLORS_3D],
        "patients": [],
    }

    processed = 0
    for pid in test_ids:
        if args.limit is not None and processed >= args.limit:
            break
        pdir = args.pddca_root / pid
        if not (pdir / "img.nrrd").is_file():
            print("Atlanıyor (img.nrrd yok):", pid)
            continue
        print("İşleniyor:", pid)
        img_vol, gt = load_pddca_patient(pid, args.pddca_root)
        pred = run_inference(img_vol, model, device)

        sub = args.out / pid
        save_orthogonal_panel(
            sub / "gt.png",
            f"PDDCA — Ground truth ({pid})",
            orthogonal_rgb_views(img_vol, gt, pred, "gt"),
        )
        save_orthogonal_panel(
            sub / "pred.png",
            f"PDDCA — Model tahmini ({pid})",
            orthogonal_rgb_views(img_vol, gt, pred, "pred"),
        )
        save_orthogonal_panel(
            sub / "fn.png",
            f"Kaçırılan bölgeler — GT'de var, tahminde yok ({pid})",
            orthogonal_rgb_views(img_vol, gt, pred, "fn"),
        )

        om = compute_organ_metrics(gt, pred)
        entry = {
            "id": pid,
            "gt": f"pddca-viz/{pid}/gt.png",
            "pred": f"pddca-viz/{pid}/pred.png",
            "fn": f"pddca-viz/{pid}/fn.png",
            "scene3d": None,
            "organMetrics": om,
            "dicePercent": {k: v["dicePercent"] for k, v in om.items()},
        }
        if not args.no_3d and _HAS_PLOTLY_3D:
            html_path = sub / "scene_3d.html"
            if write_gt_pred_3d_html(
                gt,
                pred,
                html_path,
                pid,
                step_size=args.mesh_step,
                embed_plotlyjs=args.embed_plotlyjs,
            ):
                entry["scene3d"] = f"pddca-viz/{pid}/scene_3d.html"
                print("  → 3D:", html_path.name)
        manifest["patients"].append(entry)
        processed += 1

    with open(args.out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Tamamlandı. manifest:", args.out / "manifest.json")


if __name__ == "__main__":
    main()
