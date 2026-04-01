#!/usr/bin/env python3
"""
Tüm ground truth slice'ları hasta klasörlerine kaydeder.
Yapı: ground_truth_slices/<hasta_id>/slice_000.png, slice_001.png, ...
Her görüntü: 4 panel (CT, overlay, CT+maskeler, renk lejandı)
"""
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import nrrd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Sabitler
STRUCTURE_NAMES = [
    'BrainStem', 'Chiasm', 'Mandible',
    'OpticNerve_L', 'OpticNerve_R',
    'Parotid_L', 'Parotid_R',
    'Submandibular_L', 'Submandibular_R',
]
HANSEG_OAR_MAP = {
    'BrainStem': 'Brainstem', 'Chiasm': 'OpticChiasm', 'Mandible': 'Bone_Mandible',
    'OpticNerve_L': 'OpticNrv_L', 'OpticNerve_R': 'OpticNrv_R',
    'Parotid_L': 'Parotid_L', 'Parotid_R': 'Parotid_R',
    'Submandibular_L': 'Glnd_Submand_L', 'Submandibular_R': 'Glnd_Submand_R',
}
IMG_SIZE = 256
HU_MIN, HU_MAX = -200, 300
COLORS = [
    (1, 0.2, 0.2, 0.5), (1, 0.8, 0, 0.6), (0.2, 0.8, 0.2, 0.5),
    (0, 0.6, 1, 0.6), (0.6, 0, 1, 0.6), (1, 0.5, 0, 0.5),
    (1, 0, 0.5, 0.5), (0, 0.8, 0.8, 0.5), (0.8, 0.6, 0.2, 0.5),
]
COLOR_NAMES = ['Kırmızı', 'Sarı', 'Yeşil', 'Mavi', 'Mor', 'Turuncu', 'Pembe', 'Cyan', 'Altın']


def find_pddca_and_hanseg():
    PDDCA_DIR = None
    HANSEG_DIR = None
    base = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path(__file__).resolve().parent
    for root, dirs, files in os.walk(str(base)):
        rp = Path(root)
        if 'data_split.json' in files and PDDCA_DIR is None:
            PDDCA_DIR = rp
        if 'set_1' in dirs and HANSEG_DIR is None:
            HANSEG_DIR = rp
    for root_cand in [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path('.')]:
        if (root_cand / 'data_split.json').exists() and PDDCA_DIR is None:
            PDDCA_DIR = root_cand
        if (root_cand / 'HaN-Seg' / 'set_1').exists() and HANSEG_DIR is None:
            HANSEG_DIR = root_cand / 'HaN-Seg'
        h2 = root_cand / 'HaN-Seg' / 'HaN-Seg' / 'set_1'
        if HANSEG_DIR is None and h2.exists():
            HANSEG_DIR = root_cand / 'HaN-Seg' / 'HaN-Seg'
    return PDDCA_DIR, HANSEG_DIR


def load_pddca_volume_and_masks(patient_id, pddca_dir):
    patient_dir = pddca_dir / patient_id
    img_np, _ = nrrd.read(str(patient_dir / 'img.nrrd'))
    label = np.zeros(img_np.shape, dtype=np.uint8)
    for idx, name in enumerate(STRUCTURE_NAMES, start=1):
        p = patient_dir / 'structures' / f'{name}.nrrd'
        if p.exists():
            mask_np, _ = nrrd.read(str(p))
            label[mask_np > 0] = idx
    return img_np, label


def load_hanseg_volume_and_masks(case_dir, case_id):
    ct_path = case_dir / f'{case_id}_IMG_CT.nrrd'
    img_np, _ = nrrd.read(str(ct_path))
    label = np.zeros(img_np.shape, dtype=np.uint8)
    for idx, (_, hanseg_oar) in enumerate(HANSEG_OAR_MAP.items(), start=1):
        oar_path = case_dir / f'{case_id}_OAR_{hanseg_oar}.seg.nrrd'
        if not oar_path.exists():
            continue
        try:
            mask_np, _ = nrrd.read(str(oar_path))
            if mask_np.ndim == 4:
                mask_np = mask_np[:, :, :, 0]
            if mask_np.shape != img_np.shape:
                zf = [img_np.shape[i] / mask_np.shape[i] for i in range(3)]
                mask_np = ndi.zoom(mask_np, zf, order=0)
            label[mask_np > 0] = idx
        except Exception:
            pass
    return img_np, label


def save_ground_truth_slice(img_slice, label_slice, save_path, title='Ground Truth'):
    img_norm = np.clip(img_slice.astype(np.float32), HU_MIN, HU_MAX)
    img_norm = (img_norm - HU_MIN) / (HU_MAX - HU_MIN)
    if img_norm.shape[0] != IMG_SIZE:
        zf = IMG_SIZE / img_norm.shape[0]
        img_norm = ndi.zoom(img_norm, zf, order=1)
        label_slice = ndi.zoom(label_slice.astype(np.float32), zf, order=0).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(img_norm, cmap='gray')
    axes[0, 0].set_title('CT')
    axes[0, 0].axis('off')

    overlay = np.zeros((*img_norm.shape, 4))
    overlay[:, :, :3] = np.stack([img_norm] * 3, axis=-1)
    overlay[:, :, 3] = 1.0
    for c in range(1, 10):
        if (label_slice == c).any():
            col = COLORS[c - 1]
            mask = (label_slice == c)
            overlay[mask, 0] = col[0]
            overlay[mask, 1] = col[1]
            overlay[mask, 2] = col[2]
            overlay[mask, 3] = min(1.0, col[3] + 0.5)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title('Tum organlar (overlay)')
    axes[0, 1].axis('off')

    label_rgb = np.zeros((*label_slice.shape, 3))
    for c in range(1, 10):
        mask = (label_slice == c)
        label_rgb[mask, 0] = COLORS[c-1][0]
        label_rgb[mask, 1] = COLORS[c-1][1]
        label_rgb[mask, 2] = COLORS[c-1][2]
    axes[1, 0].imshow(img_norm, cmap='gray')
    axes[1, 0].imshow(label_rgb, alpha=0.6)
    axes[1, 0].set_title('CT + Maskeler')
    axes[1, 0].axis('off')

    axes[1, 1].axis('off')
    # Renk + organ eşlemesi (örn: "1. Kırmızı — BrainStem")
    leg_text = '\n'.join([f'{i+1}. {COLOR_NAMES[i]} — {STRUCTURE_NAMES[i]}' for i in range(9)])
    axes[1, 1].text(0.05, 0.95, 'Renk → Organ:\n' + leg_text, fontsize=9,
                   verticalalignment='top', transform=axes[1, 1].transAxes,
                   family='monospace')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    PDDCA_DIR, HANSEG_DIR = find_pddca_and_hanseg()
    OUT_GT = Path(__file__).resolve().parent / 'ground_truth_slices'
    OUT_GT.mkdir(exist_ok=True)

    # PDDCA
    if PDDCA_DIR and (PDDCA_DIR / 'data_split.json').exists():
        with open(PDDCA_DIR / 'data_split.json') as f:
            split = json.load(f)
        all_patients = (
            split.get('test_offsite', []) +
            split.get('test_onsite', []) +
            split.get('train', []) +
            split.get('train_optional', [])
        )
        seen = set()
        patient_ids = [x for x in all_patients if x not in seen and not seen.add(x)]

        for pid in patient_ids:
            pdir = PDDCA_DIR / pid
            if not (pdir / 'img.nrrd').exists():
                continue
            try:
                img, label = load_pddca_volume_and_masks(pid, PDDCA_DIR)
            except Exception as e:
                print(f"[HATA] {pid}: {e}")
                continue
            n_slices = img.shape[2]
            patient_out = OUT_GT / pid
            patient_out.mkdir(exist_ok=True)
            print(f"PDDCA {pid}: {n_slices} slice...")
            for s in range(n_slices):
                save_path = patient_out / f"slice_{s:03d}.png"
                save_ground_truth_slice(
                    img[:, :, s], label[:, :, s],
                    save_path,
                    title=f'{pid} — Slice {s}/{n_slices}'
                )
            print(f"  → {patient_out}")

    print(f"Tamamlandı: {OUT_GT.absolute()}")


if __name__ == '__main__':
    main()
