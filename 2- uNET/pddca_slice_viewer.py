"""
2D axial kesit görüntüleyici HTML üretici.
Gereken: Pillow, numpy. Torch veya Plotly gerekmez.

Üretilen HTML:
  - 2 ya da 3 panel: HU | GT overlay | Tahmin overlay
  - Axial (Z) kaydırıcı
  - Her kesit WebP olarak base64'e gömülü → tek dosya, bağımsız çalışır
"""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np

# Label index → organ adı (pddca_plotly_3d.STRUCTURE_NAMES ile aynı sıra, 1-indexed)
ORGAN_NAMES = {
    1: "BrainStem",
    2: "Chiasm",
    3: "Mandible",
    4: "OpticNerve_L",
    5: "OpticNerve_R",
    6: "Parotid_L",
    7: "Parotid_R",
    8: "Submandibular_L",
    9: "Submandibular_R",
}

# RGBA renkleri (0-255). pddca_plotly_3d.COLORS_3D ile uyumlu.
ORGAN_RGBA: dict[int, tuple[int, int, int, int]] = {
    1: (255,  51,  51, 210),   # BrainStem       — kırmızı
    2: (255, 204,   0, 210),   # Chiasm           — sarı
    3:  (51, 204,  51, 210),   # Mandible         — yeşil
    4:   (0, 153, 255, 210),   # OpticNerve_L     — mavi
    5: (153,   0, 255, 210),   # OpticNerve_R     — mor
    6: (255, 128,   0, 210),   # Parotid_L        — turuncu
    7: (255,   0, 128, 210),   # Parotid_R        — pembe-kırmızı
    8:   (0, 204, 204, 210),   # Submandibular_L  — camgöbeği
    9: (204, 153,  51, 210),   # Submandibular_R  — kahve-altın
}

# HU penceresi: yumuşak doku (soft tissue window)
HU_LOW  = -160
HU_HIGH =  240


def _hu_to_gray(hu_slice: np.ndarray) -> np.ndarray:
    """HU değerlerini 0-255 gri tonlamasına çevirir."""
    clipped = np.clip(hu_slice.astype(np.float32), HU_LOW, HU_HIGH)
    gray = ((clipped - HU_LOW) / (HU_HIGH - HU_LOW) * 255).astype(np.uint8)
    return gray


def _label_overlay(gray: np.ndarray, label_slice: np.ndarray) -> np.ndarray:
    """Gri HU üzerine renkli organ etiketleri bindirme. RGBA → RGB döndürür."""
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for lbl, (r, g, b, a) in ORGAN_RGBA.items():
        mask = label_slice == lbl
        if not mask.any():
            continue
        alpha = a / 255.0
        rgb[mask, 0] = rgb[mask, 0] * (1 - alpha) + r * alpha
        rgb[mask, 1] = rgb[mask, 1] * (1 - alpha) + g * alpha
        rgb[mask, 2] = rgb[mask, 2] * (1 - alpha) + b * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _to_b64_webp(arr: np.ndarray, quality: int = 72) -> str:
    """numpy uint8 RGB array → base64 WebP string."""
    from PIL import Image  # type: ignore
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=4)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_slice_viewer_html(
    img_vol: np.ndarray,
    gt_vol: np.ndarray,
    out_html: Path,
    patient_id: str,
    pred_vol: np.ndarray | None = None,
    organ_metrics: dict | None = None,
    step: int = 1,
    quality: int = 72,
) -> bool:
    """
    Axial kesit görüntüleyici HTML üretir.

    Parametreler
    ----------
    img_vol   : (H, W, D) float — HU değerleri
    gt_vol    : (H, W, D) int  — etiket indeksleri (0 = arka plan)
    out_html  : çıktı dosyası
    patient_id: başlık için kullanılır
    pred_vol  : isteğe bağlı tahmin hacmi, None → 2 panel
    step      : her kaç kesit bir dahil edilecek (1 = hepsi, 2 = atlamalı)
    quality   : WebP kalitesi (50-85 arası iyi denge)
    """
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Pillow kurulu değil: pip install Pillow")
        return False

    D = img_vol.shape[2]
    z_indices = list(range(0, D, step))

    panels_hu:   list[str] = []
    panels_gt:   list[str] = []
    panels_pred: list[str] = []
    slice_stats: list[dict] = []

    for z in z_indices:
        hu_sl    = img_vol[:, :, z]
        gt_sl    = gt_vol[:, :, z]
        pred_sl  = pred_vol[:, :, z] if pred_vol is not None else None
        gray     = _hu_to_gray(hu_sl)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)

        panels_hu.append(_to_b64_webp(gray_rgb, quality))
        panels_gt.append(_to_b64_webp(_label_overlay(gray, gt_sl), quality))

        if pred_sl is not None:
            panels_pred.append(_to_b64_webp(_label_overlay(gray, pred_sl), quality))

        # Per-kesit istatistikler
        z_stats: dict[str, dict] = {}
        for lbl, name in ORGAN_NAMES.items():
            gt_mask  = gt_sl == lbl
            gt_n     = int(gt_mask.sum())
            if pred_sl is not None:
                pred_mask = pred_sl == lbl
                tp = int((gt_mask & pred_mask).sum())
                fn = int((gt_mask & ~pred_mask).sum())
                fp = int((~gt_mask & pred_mask).sum())
                pred_n = int(pred_mask.sum())
            else:
                tp = fn = fp = pred_n = None
            if gt_n > 0 or (pred_n or 0) > 0:
                z_stats[name] = {
                    'gt': gt_n, 'tp': tp, 'fn': fn, 'fp': fp, 'pred': pred_n
                }
        slice_stats.append(z_stats)

    has_pred = pred_vol is not None

    # Renk açıklaması (legend)
    present_labels = set(int(v) for v in np.unique(gt_vol) if v > 0)
    if pred_vol is not None:
        present_labels |= set(int(v) for v in np.unique(pred_vol) if v > 0)

    legend_items = []
    for lbl in sorted(present_labels):
        if lbl in ORGAN_RGBA and lbl in ORGAN_NAMES:
            r, g, b, _ = ORGAN_RGBA[lbl]
            name = ORGAN_NAMES[lbl]
            legend_items.append(
                f'<span class="leg-item"><span class="leg-dot" '
                f'style="background:rgb({r},{g},{b})"></span>{name}</span>'
            )
    legend_html = "\n".join(legend_items)

    # Voxel istatistik tablosu
    stats_rows = []
    if organ_metrics:
        for lbl in sorted(present_labels):
            name = ORGAN_NAMES.get(lbl, f"Label {lbl}")
            m = organ_metrics.get(name)
            if not m:
                continue
            r, g, b, _ = ORGAN_RGBA.get(lbl, (180, 180, 180, 200))
            dot = f'<span class="leg-dot" style="background:rgb({r},{g},{b});margin-right:4px"></span>'
            gt_v   = m.get("gtVoxels", "—")
            pred_v = m.get("predVoxels", "—")
            tp_v   = m.get("tpVoxels", "—")
            fn_v   = m.get("fnVoxels", "—")
            fp_v   = m.get("fpVoxels", "—")
            dice   = m.get("dicePercent", None)
            dice_s = f"{dice:.1f}%" if dice is not None else "—"
            stats_rows.append(
                f"<tr><td>{dot}{name}</td>"
                f'<td class="num">{gt_v:,}</td>'
                f'<td class="num tp">{tp_v:,}</td>'
                f'<td class="num fn">{fn_v:,}</td>'
                f'<td class="num fp">{fp_v:,}</td>'
                f'<td class="num">{pred_v:,}</td>'
                f'<td class="num dice">{dice_s}</td></tr>'
                if isinstance(gt_v, int) else
                f"<tr><td>{dot}{name}</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>"
            )

    stats_table_html = ""  # tablo artık React tarafında render ediliyor

    # Renk haritası (JS için)
    organ_colors_js = {name: f"rgb({r},{g},{b})"
                       for lbl, (r, g, b, _) in ORGAN_RGBA.items()
                       for name in [ORGAN_NAMES.get(lbl, "")] if name}

    panels_json = json.dumps({
        "hu":         panels_hu,
        "gt":         panels_gt,
        "pred":       panels_pred if has_pred else None,
        "zIndices":   z_indices,
        "hasPred":    has_pred,
        "patientId":  patient_id,
        "sliceStats": slice_stats,
        "organColors": organ_colors_js,
    })

    n_cols = 3 if has_pred else 2

    pred_panel_html = (
        "<div class='panel'><div class='panel-title'>Tahmin</div>"
        "<img id='img-pred' alt='Tahmin'/></div>"
    ) if has_pred else (
        "<div class='panel'><div class='panel-title'>Tahmin</div>"
        "<div class='no-pred'>Tahmin verisi üretiliyor…</div></div>"
    )

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<title>2D Kesit — {patient_id}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#f8fafc;
  padding:0.6rem 0.7rem;display:flex;flex-direction:column;gap:0.55rem}}
.top-row{{display:flex;align-items:center;gap:0.7rem;flex-wrap:wrap}}
.title{{font-size:0.82rem;font-weight:700;opacity:0.85;white-space:nowrap}}
#slice-slider{{flex:1;min-width:120px;accent-color:#60a5fa;cursor:pointer}}
#slice-label{{font-size:0.72rem;color:#94a3b8;white-space:nowrap}}
.panels{{display:grid;grid-template-columns:repeat({n_cols},1fr);gap:0.4rem}}
.panel{{background:#1e293b;border-radius:6px;overflow:hidden}}
.panel-title{{font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
  padding:0.3rem 0.5rem;background:#0f172a;color:#64748b;text-align:center}}
.panel img{{display:block;width:100%;height:auto;image-rendering:pixelated}}
.no-pred{{display:flex;align-items:center;justify-content:center;height:100px;
  font-size:0.68rem;color:#475569;text-align:center;padding:0.75rem}}
.legend{{display:flex;flex-wrap:wrap;gap:0.25rem 0.6rem}}
.leg-item{{display:flex;align-items:center;gap:0.25rem;font-size:0.62rem;opacity:0.75}}
.leg-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
/* stats table */
.stats-box{{background:#1e293b;border-radius:7px;overflow:hidden}}
.stats-hdr{{display:grid;grid-template-columns:1fr repeat(6,auto);gap:0;
  font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
  color:#475569;padding:0.3rem 0.5rem;border-bottom:1px solid #334155}}
.stats-hdr span{{text-align:right;padding:0 0.4rem}}
.stats-hdr span:first-child{{text-align:left;padding:0}}
#stats-body{{}}
.stat-row{{display:grid;grid-template-columns:1fr repeat(6,auto);gap:0;
  font-size:0.68rem;padding:0.22rem 0.5rem;border-bottom:1px solid rgba(255,255,255,0.04);
  align-items:center;transition:background .1s}}
.stat-row:last-child{{border-bottom:none}}
.stat-row:hover{{background:rgba(255,255,255,0.03)}}
.stat-row span{{text-align:right;padding:0 0.4rem;font-variant-numeric:tabular-nums;color:#94a3b8}}
.stat-row span:first-child{{text-align:left;padding:0;display:flex;align-items:center;gap:0.3rem;color:#cbd5e1;font-weight:500}}
.s-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0;display:inline-block}}
.tp{{color:#4ade80}}
.fn{{color:#fb923c}}
.fp{{color:#60a5fa}}
.dc{{color:#e2e8f0;font-weight:700}}
.empty-row{{padding:0.5rem;font-size:0.68rem;color:#475569;text-align:center}}
</style>
</head>
<body>
<div class="top-row">
  <span class="title">2D Kesit — {patient_id}</span>
  <input type="range" id="slice-slider" min="0" max="{len(z_indices)-1}" value="{len(z_indices)//2}" step="1"/>
  <span id="slice-label"></span>
</div>
<div class="panels">
  <div class="panel">
    <div class="panel-title">Hounsfield Unit (BT)</div>
    <img id="img-hu" alt="HU"/>
  </div>
  <div class="panel">
    <div class="panel-title">Ground Truth</div>
    <img id="img-gt" alt="GT"/>
  </div>
  {pred_panel_html}
</div>
<div class="legend">{legend_html}</div>
<div class="stats-box">
  <div class="stats-hdr">
    <span>Organ</span>
    <span>GT</span>
    <span class="tp">TP</span>
    <span class="fn">FN</span>
    <span class="fp">FP</span>
    <span>Tahmin</span>
    <span>Dice</span>
  </div>
  <div id="stats-body"></div>
</div>
<script>
const DATA = {panels_json};
const slider   = document.getElementById('slice-slider');
const sliceLabel = document.getElementById('slice-label');
const imgHu    = document.getElementById('img-hu');
const imgGt    = document.getElementById('img-gt');
const imgPred  = document.getElementById('img-pred');
const statsBody = document.getElementById('stats-body');
const total    = DATA.zIndices.length;

function fmt(n) {{ return n == null ? '—' : n.toLocaleString('tr-TR'); }}
function dice(tp, fn, fp) {{
  if (tp == null) return null;
  const d = 2*tp + fn + fp;
  return d === 0 ? null : (2*tp / d * 100);
}}

function renderStats(idx) {{
  const stats = DATA.sliceStats[idx];
  const organs = Object.keys(stats || {{}});
  if (!organs.length) {{
    statsBody.innerHTML = '<div class="empty-row">Bu kesitte etiketli alan yok.</div>';
    return;
  }}
  statsBody.innerHTML = organs.map(name => {{
    const m = stats[name];
    const clr = DATA.organColors[name] || '#888';
    const d = dice(m.tp, m.fn, m.fp);
    const dStr = d != null ? d.toFixed(1)+'%' : '—';
    return `<div class="stat-row">
      <span><span class="s-dot" style="background:${{clr}}"></span>${{name}}</span>
      <span>${{fmt(m.gt)}}</span>
      <span class="tp">${{fmt(m.tp)}}</span>
      <span class="fn">${{fmt(m.fn)}}</span>
      <span class="fp">${{fmt(m.fp)}}</span>
      <span>${{fmt(m.pred)}}</span>
      <span class="dc">${{dStr}}</span>
    </div>`;
  }}).join('');
}}

function update(idx) {{
  const z = DATA.zIndices[idx];
  imgHu.src = 'data:image/webp;base64,' + DATA.hu[idx];
  imgGt.src = 'data:image/webp;base64,' + DATA.gt[idx];
  if (imgPred && DATA.hasPred) {{
    imgPred.src = 'data:image/webp;base64,' + DATA.pred[idx];
  }}
  sliceLabel.textContent = 'Kesit ' + z + ' / ' + DATA.zIndices[total-1];
  renderStats(idx);
}}

slider.addEventListener('input', () => update(parseInt(slider.value)));
update(parseInt(slider.value));
</script>
</body>
</html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return True
