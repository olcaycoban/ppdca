#!/usr/bin/env python3
"""
Mevcut scene_3d.html (4-panel) dosyalarından yalnızca 'tahmin' panelini
(scene2) çıkarıp pred_3d.html olarak kaydeder.
İnference gerekmez — her hasta ~2-3 saniye sürer.

Kullanım:
  .venv/bin/python "2- uNET/extract_pred_3d.py" --pddca-root .
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def _extract_traces(html_text: str, target_scene: str) -> list[dict]:
    """Plotly HTML içinden verilen scene adına ait trace dict listesini döndürür."""
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', html_text, re.DOTALL)
    # En uzun script → Plotly data içeriyor
    big = max(scripts, key=len)
    idx = big.find('Plotly.newPlot(')
    if idx == -1:
        return []
    arr_start = big.index('[{', idx)

    # String-safe bracket counter
    depth = 0
    in_str = False
    escape = False
    end = arr_start
    for i, ch in enumerate(big[arr_start:], start=arr_start):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end = i
                break

    all_traces = json.loads(big[arr_start:end + 1])
    return [tr for tr in all_traces if tr.get('scene') == target_scene]


PANEL_CONFIGS = {
    'pred': {'scene_key': 'scene2', 'title': 'U-Net tahmini (3D)', 'title_prefix': 'U-Net tahmini'},
    'fn':   {'scene_key': 'scene3', 'title': 'False Negative (3D)', 'title_prefix': 'U-Net FN'},
    'fp':   {'scene_key': 'scene4', 'title': 'False Positive (3D)', 'title_prefix': 'U-Net FP'},
}


def extract_panel_3d(
    scene_html: Path,
    out_html: Path,
    patient_id: str,
    panel: str = 'pred',
) -> bool:
    if not _HAS_PLOTLY:
        print('plotly kurulu değil', file=sys.stderr)
        return False

    cfg = PANEL_CONFIGS[panel]
    text = scene_html.read_text(encoding='utf-8')
    traces = _extract_traces(text, cfg['scene_key'])
    if not traces:
        print(f'  {cfg["scene_key"]} verisi bulunamadı:', scene_html.name)
        return False

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scene'}]],
        subplot_titles=(cfg['title'],),
    )
    fig.update_layout(
        title_text=f'{cfg["title_prefix"]} — {patient_id}',
        height=520,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='white',
        scene=dict(
            xaxis=dict(title='X', backgroundcolor='white'),
            yaxis=dict(title='Y', backgroundcolor='white'),
            zaxis=dict(title='Z (kesit)', backgroundcolor='white'),
            bgcolor='#f8f9fa',
            aspectmode='data',
        ),
        showlegend=False,
    )
    for tr_dict in traces:
        tr = dict(tr_dict)
        tr.pop('scene', None)
        tr.pop('subplot', None)
        trace_type = tr.pop('type', 'mesh3d')
        if trace_type == 'mesh3d':
            fig.add_trace(go.Mesh3d(**tr))
        else:
            fig.add_trace(go.Scatter3d(**tr))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_html),
        include_plotlyjs='cdn',
        config=dict(scrollZoom=True, displaylogo=False),
        full_html=True,
    )
    return True


# Keep backward-compat alias
def extract_pred_3d(scene_html: Path, pred_html: Path, patient_id: str) -> bool:
    return extract_panel_3d(scene_html, pred_html, patient_id, panel='pred')


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--pddca-root', type=Path, default=_REPO)
    ap.add_argument(
        '--viz-dir',
        type=Path,
        default=_REPO / 'site' / 'public' / 'pddca-viz',
    )
    args = ap.parse_args()

    manifest_path = args.viz_dir / 'manifest.json'
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)

    PANELS = [
        ('pred', 'pred3d', 'pred_3d.html'),
        ('fn',   'fn3d',   'fn_3d.html'),
        ('fp',   'fp3d',   'fp_3d.html'),
    ]

    changed = False
    for p in manifest['patients']:
        pid = p['id']
        scene_html = args.viz_dir / pid / 'scene_3d.html'
        if not scene_html.is_file():
            print('  scene_3d.html yok, atlandı:', pid)
            continue
        for panel, manifest_key, filename in PANELS:
            out_html = args.viz_dir / pid / filename
            if extract_panel_3d(scene_html, out_html, pid, panel=panel):
                p[manifest_key] = f'pddca-viz/{pid}/{filename}'
                print(f'  OK {panel} 3D: {pid}')
                changed = True
            else:
                print(f'  BAŞARISIZ {panel}: {pid}')

    if changed:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print('manifest güncellendi.')
    else:
        print('Değişiklik yok.')


if __name__ == '__main__':
    main()
