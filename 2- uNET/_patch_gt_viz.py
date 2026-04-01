#!/usr/bin/env python3
"""Patch ground_truth_visualization.ipynb: add display param + save-all-slices to folder."""
import json
from pathlib import Path

nb_path = Path(__file__).parent / 'ground_truth_visualization.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update show_ground_truth: add display=True, and conditional show/close
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if "def show_ground_truth(img_slice, label_slice, title='Ground Truth', save_path=None):" in src and "axes[1, 1].text" in src:
        new_src = src.replace(
            "def show_ground_truth(img_slice, label_slice, title='Ground Truth', save_path=None):",
            "def show_ground_truth(img_slice, label_slice, title='Ground Truth', save_path=None, display=True):"
        )
        new_src = new_src.replace(
            "    if save_path:\n        plt.savefig(save_path, dpi=150, bbox_inches='tight')\n    plt.show()",
            "    if save_path:\n        plt.savefig(save_path, dpi=150, bbox_inches='tight')\n    if display:\n        plt.show()\n    else:\n        plt.close(fig)"
        )
        nb['cells'][i]['source'] = new_src.splitlines(True)
        break

# 2. Find and replace the 4b section (Tüm slice'lar — Grid) with save-to-folder version
NEW_MD = """## 4b. Tüm slice'ları klasöre kaydet

Her hasta için klasör açılır, her slice 4 panelli ground truth formatında PNG olarak kaydedilir.
Yapı: `ground_truth_slices/<hasta_id>/slice_000.png`, `slice_001.png`, ...
"""

NEW_CODE = """# Tüm slice'ları hasta klasörüne kaydet (4 panelli format)
OUT_GT = Path('ground_truth_slices')  # Çıktı klasörü
OUT_GT.mkdir(exist_ok=True)

# PDDCA test hastaları
with open(PDDCA_DIR / 'data_split.json') as f:
    split = json.load(f)
PATIENT_IDS = split.get('test_offsite', split.get('test', []))[:3]  # İlk 3 hasta (hepsi için [] kaldır)

for pid in PATIENT_IDS:
    pdir = PDDCA_DIR / pid
    if not (pdir / 'img.nrrd').exists():
        continue
    img, label = load_pddca_volume_and_masks(pid)
    n_slices = img.shape[2]
    patient_dir = OUT_GT / pid
    patient_dir.mkdir(exist_ok=True)
    print(f"{pid}: {n_slices} slice kaydediliyor...")
    for s in range(n_slices):
        save_path = patient_dir / f"slice_{s:03d}.png"
        show_ground_truth(img[:, :, s], label[:, :, s],
                         title=f'{pid} — Slice {s}/{n_slices}',
                         save_path=save_path, display=False)
    print(f"  → {patient_dir}")

# HaN-Seg hastaları (ilk 3)
set_dir = HANSEG_DIR / 'set_1'
cases = sorted([d.name for d in set_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])[:3]
for cid in cases:
    case_dir = set_dir / cid
    if not (case_dir / f'{cid}_IMG_CT.nrrd').exists():
        continue
    img, label = load_hanseg_volume_and_masks(case_dir, cid)
    n_slices = img.shape[2]
    patient_dir = OUT_GT / cid
    patient_dir.mkdir(exist_ok=True)
    print(f"{cid}: {n_slices} slice kaydediliyor...")
    for s in range(n_slices):
        save_path = patient_dir / f"slice_{s:03d}.png"
        show_ground_truth(img[:, :, s], label[:, :, s],
                         title=f'{cid} — Slice {s}/{n_slices}',
                         save_path=save_path, display=False)
    print(f"  → {patient_dir}")

print(f"Tamamlandı: {OUT_GT.absolute()}")
"""

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell.get('source', []))
        if "## 4b. Tüm slice'lar" in src and "Grid" in src:
            nb['cells'][i]['source'] = NEW_MD.strip().splitlines(True)
            break

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        if "SLICES_PER_PAGE = 20" in src and "Tüm slice'ları göster" in src:
            nb['cells'][i]['source'] = NEW_CODE.strip().splitlines(True)
            nb['cells'][i]['outputs'] = []
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Notebook güncellendi.")
