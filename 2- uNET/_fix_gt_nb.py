#!/usr/bin/env python3
"""Fix ground_truth_visualization.ipynb: clear outputs, add save-all-slices section."""
import json
import sys
from pathlib import Path

nb_path = Path(__file__).parent / 'ground_truth_visualization.ipynb'
if not nb_path.exists():
    print("Notebook not found")
    sys.exit(1)

# Load with strict=False to tolerate minor issues, or try reading and fixing
with open(nb_path, 'r') as f:
    raw = f.read()

# Try to load - if fails, try to fix by removing truncated outputs
try:
    nb = json.loads(raw)
except json.JSONDecodeError as e:
    print(f"JSON error: {e}")
    # Try clearing all outputs as last resort
    nb = json.loads(raw[:e.pos] + '"}')  # won't work for complex cases
    sys.exit(1)

# Clear all outputs
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# Find show_ground_truth and add display param
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = cell.get('source', [])
    if isinstance(src, list):
        src_str = ''.join(src)
    else:
        src_str = src
    if 'def show_ground_truth(img_slice' in src_str and 'display=True' not in src_str:
        new_src = src_str.replace(
            "def show_ground_truth(img_slice, label_slice, title='Ground Truth', save_path=None):",
            "def show_ground_truth(img_slice, label_slice, title='Ground Truth', save_path=None, display=True):"
        )
        new_src = new_src.replace(
            """    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()""",
            """    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if display:
        plt.show()
    else:
        plt.close(fig)"""
        )
        cell['source'] = new_src.splitlines(True)
        cell['source'] = [s if s.endswith('\n') else s + '\n' for s in cell['source']]
        break

# Find 4b section and replace with save-to-folder version
SAVE_CELL = '''
# Tüm slice'ları hasta hasta, slice slice klasöre kaydet
OUT_ROOT = Path('ground_truth_slices')  # ana klasör
OUT_ROOT.mkdir(exist_ok=True)

# PDDCA için test_offsite hastaları
with open(PDDCA_DIR / 'data_split.json') as f:
    split = json.load(f)
PATIENT_IDS = split.get('test_offsite', split.get('test', []))[:3]  # ilk 3 hasta (tümü için [:10])

for pid in PATIENT_IDS:
    if not (PDDCA_DIR / pid / 'img.nrrd').exists():
        continue
    img, label = load_pddca_volume_and_masks(pid)
    out_dir = OUT_ROOT / pid
    out_dir.mkdir(exist_ok=True)
    n_slices = img.shape[2]
    for s in range(n_slices):
        save_path = out_dir / f'slice_{s:04d}.png'
        show_ground_truth(img[:, :, s], label[:, :, s],
                         title=f'{pid} — Slice {s}/{n_slices}',
                         save_path=save_path, display=False)
    print(f'{pid}: {n_slices} slice kaydedildi -> {out_dir}')

print(f"Tamam. Klasör: {OUT_ROOT}")
'''

for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'markdown':
        src = ''.join(cell.get('source', []))
        if "4b. Tüm slice'lar" in src and "Grid" in src:
            cell['source'] = ["## 4b. Tüm slice'ları klasöre kaydet\n\nHer hasta için ayrı klasör, her slice için ayrı PNG (4 panel formatında).\n"]
            break

for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'SLICES_PER_PAGE' in src and 'page_start' in src:
            cell['source'] = [line + '\n' for line in SAVE_CELL.strip().split('\n')]
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Notebook güncellendi.")