"""
Plotly 3D sahne üretimi (torch gerekmez). U-Net export ve atlas pickle export tarafından kullanılır.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import nrrd
import scipy.ndimage as ndi

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
    method_label: str | None = None,
) -> bool:
    if not _HAS_PLOTLY_3D:
        return False
    disagree = gt_label_vol != pred_label_vol
    fn_vol = np.where((gt_label_vol > 0) & disagree, gt_label_vol, 0).astype(np.uint8)
    fp_vol = np.where((pred_label_vol > 0) & disagree, pred_label_vol, 0).astype(np.uint8)

    pred_panel_label = f"{method_label} tahmini (3D)" if method_label else "Model tahmini (3D)"

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "Ground truth (3D)",
            pred_panel_label,
            "GT'de olup tahminle uyuşmayan (FN / yanlış sınıf)",
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
    base_title = f"PDDCA — 3D organ yüzeyleri ({patient_id})"
    full_title = f"{method_label} — {patient_id}" if method_label else base_title
    fig.update_layout(
        title_text=full_title,
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
        hover_panel=pred_panel_label,
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


def write_gt_only_3d_html(
    gt_label_vol: np.ndarray,
    out_html: Path,
    patient_id: str,
    step_size: int = 2,
    embed_plotlyjs: bool = False,
) -> bool:
    """GT-only tek panel 3B sahne (model gerekmez)."""
    if not _HAS_PLOTLY_3D:
        return False

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scene"}]],
        subplot_titles=("Ground truth (3D)",),
    )
    scene_kw = dict(
        xaxis=dict(title="X", backgroundcolor="white"),
        yaxis=dict(title="Y", backgroundcolor="white"),
        zaxis=dict(title="Z (kesit)", backgroundcolor="white"),
        bgcolor="#f8f9fa",
        aspectmode="data",
    )
    fig.update_layout(
        title_text=f"Ground truth — {patient_id}",
        height=520,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor="white",
        scene=scene_kw,
        showlegend=False,
    )
    for tr in mesh3d_traces_for_volume(
        gt_label_vol,
        step_size=step_size,
        min_voxels=10,
        hover_panel="Ground truth (3D)",
    ):
        fig.add_trace(tr, row=1, col=1)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    include_js = True if embed_plotlyjs else "cdn"
    fig.write_html(
        str(out_html),
        include_plotlyjs=include_js,
        config=dict(scrollZoom=True, displaylogo=False),
        full_html=True,
    )
    return True


def write_pred_only_3d_html(
    pred_label_vol: np.ndarray,
    out_html: Path,
    patient_id: str,
    method_label: str | None = None,
    step_size: int = 2,
    embed_plotlyjs: bool = False,
) -> bool:
    """Yalnızca atlas tahmini tek panel (GT ve FN/FP yok)."""
    if not _HAS_PLOTLY_3D:
        return False

    panel_label = f"{method_label} tahmini (3D)" if method_label else "Atlas tahmini (3D)"
    title = f"{method_label} — {patient_id}" if method_label else f"Atlas tahmini — {patient_id}"

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scene"}]],
        subplot_titles=(panel_label,),
    )
    scene_kw = dict(
        xaxis=dict(title="X", backgroundcolor="white"),
        yaxis=dict(title="Y", backgroundcolor="white"),
        zaxis=dict(title="Z (kesit)", backgroundcolor="white"),
        bgcolor="#f8f9fa",
        aspectmode="data",
    )
    fig.update_layout(
        title_text=title,
        height=520,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor="white",
        scene=scene_kw,
        showlegend=False,
    )
    for tr in mesh3d_traces_for_volume(
        pred_label_vol,
        step_size=step_size,
        min_voxels=10,
        hover_panel=panel_label,
    ):
        fig.add_trace(tr, row=1, col=1)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    include_js = True if embed_plotlyjs else "cdn"
    fig.write_html(
        str(out_html),
        include_plotlyjs=include_js,
        config=dict(scrollZoom=True, displaylogo=False),
        full_html=True,
    )
    return True


def write_fn_fp_pair_3d_html(
    gt_label_vol: np.ndarray,
    pred_label_vol: np.ndarray,
    out_html: Path,
    patient_id: str,
    method_title: str,
    embed_plotlyjs: bool = False,
) -> bool:
    """Yalnızca alt satır: FN | FP (ekran görüntüsüyle aynı iki panel)."""
    if not _HAS_PLOTLY_3D:
        return False
    disagree = gt_label_vol != pred_label_vol
    fn_vol = np.where((gt_label_vol > 0) & disagree, gt_label_vol, 0).astype(np.uint8)
    fp_vol = np.where((pred_label_vol > 0) & disagree, pred_label_vol, 0).astype(np.uint8)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "GT'de olup tahminle uyuşmayan (FN / yanlış sınıf)",
            "Tahminde olup GT ile uyuşmayan (FP / yanlış sınıf)",
        ),
        horizontal_spacing=0.06,
    )
    scene_kw = dict(
        xaxis=dict(title="X", backgroundcolor="white"),
        yaxis=dict(title="Y", backgroundcolor="white"),
        zaxis=dict(title="Z (kesit)", backgroundcolor="white"),
        bgcolor="#f8f9fa",
        aspectmode="data",
    )
    fig.update_layout(
        title_text=f"{method_title} — {patient_id}",
        height=640,
        margin=dict(l=0, r=0, b=0, t=56),
        paper_bgcolor="white",
        scene=scene_kw,
        scene2=dict(**scene_kw),
        showlegend=False,
    )
    rng_fn = np.random.default_rng(zlib.adler32((patient_id + "|fn|" + method_title).encode()) & 0xFFFFFFFF)
    rng_fp = np.random.default_rng(zlib.adler32((patient_id + "|fp|" + method_title).encode()) & 0xFFFFFFFF)
    for tr in traces_for_error_subplot(
        fn_vol,
        (0.95, 0.25, 0.35),
        rng_fn,
        hover_panel="GT ≠ tahmin (GT etiketi)",
    ):
        fig.add_trace(tr, row=1, col=1)
    for tr in traces_for_error_subplot(
        fp_vol,
        (0.95, 0.55, 0.1),
        rng_fp,
        hover_panel="Tahmin ≠ GT (tahmin etiketi)",
    ):
        fig.add_trace(tr, row=1, col=2)

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
