#!/usr/bin/env python3
"""notebook-5_hanseg9organs_pddca10.ipynb -> 3 preprocessing reçetesi varyantları."""
import json
import copy
from pathlib import Path

BASE = Path(__file__).parent
SRC = BASE / "notebook-5_hanseg9organs_pddca10.ipynb"

with open(SRC, "r", encoding="utf-8") as f:
    nb_base = json.load(f)


def join_src(cell):
    return "".join(cell.get("source", []))


def set_src(cell, text: str):
    lines = text.split("\n")
    cell["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


# --- Reçete 1: markdown başlık ---
MD_R1 = """# U-Net — Reçete 1: nnU-Net Tarzı Ön İşleme (HaN-Seg 9 organ → PDDCA 10)

**Kaynak:** PDDCA preprocessing reçeteleri belgesi — *Altın standart* (izotropik örnekleme, persentil windowing, vücut-Z-score, elastik augment).

**Bu notebook `notebook-5_hanseg9organs_pddca10.ipynb` ile aynı pipeline’dır**; fark sadece `preprocess_volume` ve eğitim augmentasyonlarındadır:

1. **Resampling:** NRRD header’dan voxel spacing ⇒ yaklaşık **1.5 mm** izotropik (görüntü bilinear, maske nearest).
2. **Windowing:** Vücut içi piksellerde **0.5–99.5 persentil** ile kırpma.
3. **Normalizasyon:** Vücut maskesi (HU > eşik) üzerinde **Z-score**; arka plan ≈ 0.
4. **Augment:** Mevcut flip/rotate/scale + **2D elastik deformasyon** (hafif).

**Çıktı dosyaları:** `*_recipe1.*` (cache, checkpoint, Excel) — diğer deneylerle karışmaması için.
"""

MD_R2 = """# U-Net — Reçete 2: Yumuşak Doku Penceresi (HaN-Seg 9 organ → PDDCA 10)

**Kaynak:** PDDCA preprocessing reçeteleri — *Yumuşak doku odaklı* (parotis / submandibular).

**Farklar:**
1. **Sabit HU penceresi:** **[-160, 240] HU** (hard clip).
2. **Normalizasyon:** **[0, 1] Min–Max** (pencere içi).
3. İzotropik örnekleme yok (hızlı 2D baseline); istenirse config’e `ISOTROPIC_MM` eklenebilir.

**Not:** Belgedeki 96³ 3D patch eğitimi **tam 3D model** gerektirir; bu notebook 2D slice U-Net olduğu için **256×256 slice** akışı korunmuştur.

**Çıktı:** `*_recipe2.*`
"""

MD_R3 = """# U-Net — Reçete 3: Kaba Kırpma + Küçük Organ Dostu Örnekleme (HaN-Seg 9 organ → PDDCA 10)

**Kaynak:** PDDCA preprocessing reçeteleri — *Küçük organ avcısı*.

**Farklar:**
1. **Kaba 3D kırpma:** HU > **-500** ile vücut bounding box + margin; gereksiz hava hacmini at.
2. **Pencere:** **[-500, 1000] HU** clip, ardından vücut üzerinde **Z-score** (görüntü yaklaşık [-3,3] sonra [0,1]’e tic).
3. **Örnekleme:** Cache oluştururken **Chiasm / Optik sinir** (sınıf 2, 4, 5) içeren kesitlere **2× ağırlık**; arka plan dilimi olasılığı düşürülür.

**Çıktı:** `*_recipe3.*`
"""

# --- Cell 2 config patches (ek satırlar, suffix) ---
CONFIG_EXTRA_R1 = """
# Reçete 1 (nnU-Net tarzı)
ISOTROPIC_MM = 1.5
PERC_LO, PERC_HI = 0.5, 99.5
HU_BODY_THRESH = -500
RECIPE_TAG = 'recipe1'
"""

CONFIG_EXTRA_R2 = """
# Reçete 2 (yumuşak doku penceresi) — HU_MIN/HU_MAX aşağıda ezilecek
RECIPE_TAG = 'recipe2'
"""

CONFIG_EXTRA_R3 = """
# Reçete 3 (kırpma + küçük organ)
COARSE_HU_THR = -500
COARSE_MARGIN = 8
WIN_LO, WIN_HI = -500, 1000
HU_BODY_THRESH = -500
RECIPE_TAG = 'recipe3'
SMALL_ORGAN_CLASSES = [2, 4, 5]  # Chiasm, Optic L/R
"""

CELL4_R1 = r'''def load_hanseg_volume_and_masks(case_dir, case_id):
    ct_path = case_dir / f'{case_id}_IMG_CT.nrrd'
    img_np, hdr = nrrd.read(str(ct_path))
    label = np.zeros(img_np.shape, dtype=np.uint8)

    for idx, (pddca_name, hanseg_oar) in enumerate(HANSEG_OAR_MAP.items(), start=1):
        oar_path = case_dir / f'{case_id}_OAR_{hanseg_oar}.seg.nrrd'
        if not oar_path.exists():
            continue
        try:
            mask_np, _ = nrrd.read(str(oar_path))
            if mask_np.ndim == 4:
                mask_np = mask_np[:, :, :, 0]
            if mask_np.shape != img_np.shape:
                zoom_f = [img_np.shape[i] / mask_np.shape[i] for i in range(3)]
                mask_np = ndi.zoom(mask_np, zoom_f, order=0)
            label[mask_np > 0] = idx
        except Exception as e:
            pass
    return img_np, label, hdr


def load_pddca_volume_and_masks(patient_id):
    patient_dir = PDDCA_DIR / patient_id
    img_np, hdr = nrrd.read(str(patient_dir / 'img.nrrd'))
    label = np.zeros(img_np.shape, dtype=np.uint8)
    for idx, name in enumerate(STRUCTURE_NAMES, start=1):
        p = patient_dir / 'structures' / f'{name}.nrrd'
        if p.exists():
            mask_np, _ = nrrd.read(str(p))
            label[mask_np > 0] = idx
    return img_np, label, hdr


def _nrrd_spacing_mm(hdr):
    if hdr is None:
        return (1.0, 1.0, 1.0)
    sd = hdr.get('space directions')
    if sd is not None:
        try:
            sd = np.array(sd, dtype=np.float64)
            if sd.shape == (3, 3):
                sp = np.linalg.norm(sd, axis=1)
                return tuple(float(sp[i]) for i in range(3))
            flat = sd.flatten()
            if flat.size >= 3:
                return tuple(float(abs(x)) for x in flat[:3])
        except Exception:
            pass
    for key in ('spacings', 'space'):
        if key in hdr and hdr[key] is not None:
            try:
                s = np.array(hdr[key], dtype=np.float64).flatten()[:3]
                return tuple(float(abs(x)) for x in s)
            except Exception:
                pass
    return (1.0, 1.0, 1.0)


def resample_isotropic(img_np, lbl_np, spacing_mm, target_mm):
    zf = tuple(max(spacing_mm[i] / target_mm, 0.2) for i in range(3))
    if all(abs(z - 1.0) < 0.03 for z in zf):
        return img_np, lbl_np
    img_r = ndi.zoom(img_np.astype(np.float32), zf, order=1)
    lbl_r = ndi.zoom(lbl_np.astype(np.float32), zf, order=0)
    return img_r, lbl_r.astype(np.uint8)


def preprocess_volume(img_np, label_np, hdr=None, target_size=IMG_SIZE):
    """Reçete 1: izotropik + persentil window + vücut Z-score + [0,1] map."""
    sp = _nrrd_spacing_mm(hdr)
    img_np, label_np = resample_isotropic(img_np, label_np, sp, ISOTROPIC_MM)
    slices_img, slices_lbl = [], []
    for s in range(img_np.shape[2]):
        sl_raw = img_np[:, :, s].astype(np.float32)
        sl_lbl = label_np[:, :, s]
        body = sl_raw > HU_BODY_THRESH
        if body.sum() > 80:
            p_lo, p_hi = np.percentile(sl_raw[body], [PERC_LO, PERC_HI])
        else:
            p_lo, p_hi = np.percentile(sl_raw, [PERC_LO, PERC_HI])
        sl = np.clip(sl_raw, p_lo, p_hi)
        if body.sum() > 80:
            mu, sigma = sl[body].mean(), sl[body].std() + 1e-6
            sl = np.where(body, (sl - mu) / sigma, 0.0)
        else:
            mu, sigma = sl.mean(), sl.std() + 1e-6
            sl = (sl - mu) / sigma
        sl = np.clip(sl, -3.0, 3.0)
        sl = (sl + 3.0) / 6.0
        if target_size != sl.shape[0]:
            zf = target_size / sl.shape[0]
            sl = ndi.zoom(sl, zf, order=1)
            sl_lbl = ndi.zoom(sl_lbl, zf, order=0)
        slices_img.append(sl.astype(np.float32))
        slices_lbl.append(sl_lbl)
    return np.array(slices_img, dtype=np.float32), np.array(slices_lbl, dtype=np.int64)

print("Yukleme + Reçete 1 preprocess hazir.")
'''

CELL4_R2 = r'''def load_hanseg_volume_and_masks(case_dir, case_id):
    ct_path = case_dir / f'{case_id}_IMG_CT.nrrd'
    img_np, hdr = nrrd.read(str(ct_path))
    label = np.zeros(img_np.shape, dtype=np.uint8)

    for idx, (pddca_name, hanseg_oar) in enumerate(HANSEG_OAR_MAP.items(), start=1):
        oar_path = case_dir / f'{case_id}_OAR_{hanseg_oar}.seg.nrrd'
        if not oar_path.exists():
            continue
        try:
            mask_np, _ = nrrd.read(str(oar_path))
            if mask_np.ndim == 4:
                mask_np = mask_np[:, :, :, 0]
            if mask_np.shape != img_np.shape:
                zoom_f = [img_np.shape[i] / mask_np.shape[i] for i in range(3)]
                mask_np = ndi.zoom(mask_np, zoom_f, order=0)
            label[mask_np > 0] = idx
        except Exception as e:
            pass
    return img_np, label, hdr


def load_pddca_volume_and_masks(patient_id):
    patient_dir = PDDCA_DIR / patient_id
    img_np, hdr = nrrd.read(str(patient_dir / 'img.nrrd'))
    label = np.zeros(img_np.shape, dtype=np.uint8)
    for idx, name in enumerate(STRUCTURE_NAMES, start=1):
        p = patient_dir / 'structures' / f'{name}.nrrd'
        if p.exists():
            mask_np, _ = nrrd.read(str(p))
            label[mask_np > 0] = idx
    return img_np, label, hdr


def preprocess_volume(img_np, label_np, hdr=None, target_size=IMG_SIZE):
    """Reçete 2: sabit HU [-160, 240] + min-max [0,1]."""
    slices_img, slices_lbl = [], []
    for s in range(img_np.shape[2]):
        sl = np.clip(img_np[:, :, s].astype(np.float32), HU_MIN, HU_MAX)
        sl = (sl - HU_MIN) / (HU_MAX - HU_MIN + 1e-6)
        sl_lbl = label_np[:, :, s]
        if target_size != sl.shape[0]:
            zf = target_size / sl.shape[0]
            sl = ndi.zoom(sl, zf, order=1)
            sl_lbl = ndi.zoom(sl_lbl, zf, order=0)
        slices_img.append(sl.astype(np.float32))
        slices_lbl.append(sl_lbl)
    return np.array(slices_img, dtype=np.float32), np.array(slices_lbl, dtype=np.int64)

print("Yukleme + Reçete 2 preprocess hazir.")
'''

CELL4_R3 = r'''def load_hanseg_volume_and_masks(case_dir, case_id):
    ct_path = case_dir / f'{case_id}_IMG_CT.nrrd'
    img_np, hdr = nrrd.read(str(ct_path))
    label = np.zeros(img_np.shape, dtype=np.uint8)

    for idx, (pddca_name, hanseg_oar) in enumerate(HANSEG_OAR_MAP.items(), start=1):
        oar_path = case_dir / f'{case_id}_OAR_{hanseg_oar}.seg.nrrd'
        if not oar_path.exists():
            continue
        try:
            mask_np, _ = nrrd.read(str(oar_path))
            if mask_np.ndim == 4:
                mask_np = mask_np[:, :, :, 0]
            if mask_np.shape != img_np.shape:
                zoom_f = [img_np.shape[i] / mask_np.shape[i] for i in range(3)]
                mask_np = ndi.zoom(mask_np, zoom_f, order=0)
            label[mask_np > 0] = idx
        except Exception as e:
            pass
    return img_np, label, hdr


def load_pddca_volume_and_masks(patient_id):
    patient_dir = PDDCA_DIR / patient_id
    img_np, hdr = nrrd.read(str(patient_dir / 'img.nrrd'))
    label = np.zeros(img_np.shape, dtype=np.uint8)
    for idx, name in enumerate(STRUCTURE_NAMES, start=1):
        p = patient_dir / 'structures' / f'{name}.nrrd'
        if p.exists():
            mask_np, _ = nrrd.read(str(p))
            label[mask_np > 0] = idx
    return img_np, label, hdr


def coarse_crop_volume(img_np, lbl_np, hu_thr=-500, margin=8):
    mask = img_np > hu_thr
    if mask.sum() == 0:
        return img_np, lbl_np
    coords = np.argwhere(mask)
    i0, j0, k0 = coords.min(axis=0)
    i1, j1, k1 = coords.max(axis=0) + 1
    i0 = max(0, i0 - margin); j0 = max(0, j0 - margin); k0 = max(0, k0 - margin)
    i1 = min(img_np.shape[0], i1 + margin)
    j1 = min(img_np.shape[1], j1 + margin)
    k1 = min(img_np.shape[2], k1 + margin)
    return img_np[i0:i1, j0:j1, k0:k1], lbl_np[i0:i1, j0:j1, k0:k1]


def preprocess_volume(img_np, label_np, hdr=None, target_size=IMG_SIZE):
    """Reçete 3: kaba crop + [-500,1000] + vücut Z-score."""
    img_np, label_np = coarse_crop_volume(img_np, label_np, COARSE_HU_THR, COARSE_MARGIN)
    slices_img, slices_lbl = [], []
    for s in range(img_np.shape[2]):
        sl_raw = img_np[:, :, s].astype(np.float32)
        sl_lbl = label_np[:, :, s]
        sl = np.clip(sl_raw, WIN_LO, WIN_HI)
        body = sl_raw > HU_BODY_THRESH
        if body.sum() > 80:
            mu, sigma = sl[body].mean(), sl[body].std() + 1e-6
            sl = np.where(body, (sl - mu) / sigma, 0.0)
        else:
            mu, sigma = sl.mean(), sl.std() + 1e-6
            sl = (sl - mu) / sigma
        sl = np.clip(sl, -3.0, 3.0)
        sl = (sl + 3.0) / 6.0
        # Kaba crop sonrasi kesitler dikdortgen olabilir; tek skala W != target verir
        h, w = sl.shape
        if h != target_size or w != target_size:
            zfh, zfw = target_size / h, target_size / w
            sl = ndi.zoom(sl, (zfh, zfw), order=1)
            sl_lbl = ndi.zoom(sl_lbl.astype(np.float32), (zfh, zfw), order=0)
            sl_lbl = np.rint(sl_lbl).astype(np.int64)
        slices_img.append(sl.astype(np.float32))
        slices_lbl.append(sl_lbl)
    return np.array(slices_img, dtype=np.float32), np.array(slices_lbl, dtype=np.int64)

print("Yukleme + Reçete 3 preprocess hazir.")
'''

ELASTIC_BLOCK = """
            if np.random.rand() < 0.15:
                sh = img.shape
                dx = ndi.gaussian_filter((np.random.rand(*sh) * 2 - 1), 4, mode='constant', cval=0) * 8
                dy = ndi.gaussian_filter((np.random.rand(*sh) * 2 - 1), 4, mode='constant', cval=0) * 8
                y, x = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing='ij')
                iy = np.clip(y + dy, 0, sh[0] - 1)
                ix = np.clip(x + dx, 0, sh[1] - 1)
                img = ndi.map_coordinates(img, [iy, ix], order=1, mode='reflect')
                lbl = ndi.map_coordinates(lbl.astype(np.float32), [iy, ix], order=0, mode='reflect')
                lbl = np.round(lbl).astype(np.int64)
"""


def patch_config_cell(cell, recipe: str):
    s = join_src(cell)
    if recipe == "r1":
        s = s.replace(
            "HU_MIN, HU_MAX = -200, 300\n",
            "# Reçete 1: persentil ile kırpma (HU_MIN/MAX sabit kullanılmaz)\nHU_MIN, HU_MAX = -200, 300  # yedek; preprocess'te persentil kullanılır\n",
        )
        s += CONFIG_EXTRA_R1
    elif recipe == "r2":
        s = s.replace("HU_MIN, HU_MAX = -200, 300", "HU_MIN, HU_MAX = -160, 240  # Reçete 2 sabit pencere")
        s += CONFIG_EXTRA_R2
    else:
        s = s.replace(
            "HU_MIN, HU_MAX = -200, 300\n",
            "# Reçete 3: pencere preprocess_volume içinde (WIN_LO/HI)\nHU_MIN, HU_MAX = -500, 1000\n",
        )
        s += CONFIG_EXTRA_R3
    set_src(cell, s.rstrip("\n"))


def patch_cache_cell(cell, tag_key: str, recipe_short: str):
    s = join_src(cell)
    s = s.replace(
        "CACHE_PATH = OUT_DIR / 'hanseg_unet_train_cache.pkl'",
        f"CACHE_PATH = OUT_DIR / 'hanseg_unet_train_cache_{tag_key}.pkl'",
    )
    s = s.replace("img, lbl, _ = load_hanseg_volume_and_masks(case_dir, cid)", "img, lbl, hdr = load_hanseg_volume_and_masks(case_dir, cid)")
    s = s.replace("sl_img, sl_lbl = preprocess_volume(img, lbl)", "sl_img, sl_lbl = preprocess_volume(img, lbl, hdr)")

    if recipe_short == "r3":  # küçük organ ağırlıklı örnekleme
        # override slice sampling: higher weight for small organ slices
        old_loop = """            for j in range(len(sl_img)):
                if sl_lbl[j].max() > 0:
                    train_slices.append(sl_img[j])
                    train_labels.append(sl_lbl[j])
                elif np.random.rand() < 0.2:
                    train_slices.append(sl_img[j])
                    train_labels.append(sl_lbl[j])"""
        new_loop = """            for j in range(len(sl_img)):
                row = sl_lbl[j]
                has_fg = row.max() > 0
                has_small = any((row == c).any() for c in SMALL_ORGAN_CLASSES)
                if has_fg:
                    train_slices.append(sl_img[j])
                    train_labels.append(sl_lbl[j])
                    if has_small and np.random.rand() < 0.5:
                        train_slices.append(sl_img[j])
                        train_labels.append(sl_lbl[j])
                elif np.random.rand() < 0.12:
                    train_slices.append(sl_img[j])
                    train_labels.append(sl_lbl[j])"""
        s = s.replace(old_loop, new_loop)

    set_src(cell, s.rstrip("\n"))


def patch_test_cell(cell):
    s = join_src(cell)
    s = s.replace("img, lbl, _ = load_pddca_volume_and_masks(pid)", "img, lbl, hdr = load_pddca_volume_and_masks(pid)")
    s = s.replace("sl_img, sl_lbl = preprocess_volume(img, lbl)", "sl_img, sl_lbl = preprocess_volume(img, lbl, hdr)")
    set_src(cell, s.rstrip("\n"))


def patch_train_cell(cell, recipe: str):
    s = join_src(cell)
    s = s.replace(
        "CKPT_PATH = OUT_DIR / 'hanseg_unet_best.pth'",
        f"CKPT_PATH = OUT_DIR / 'hanseg_unet_best_{recipe}.pth'",
    )
    set_src(cell, s.rstrip("\n"))


def patch_results_cell(cell, recipe: str):
    s = join_src(cell)
    s = s.replace(
        "out_xlsx = OUT_DIR / 'hanseg_unet_pddca10_results.xlsx'",
        f"out_xlsx = OUT_DIR / 'hanseg_unet_pddca10_results_{recipe}.xlsx'",
    )
    s = s.replace(
        "out_pkl = OUT_DIR / 'hanseg_unet_pddca10_results.pkl'",
        f"out_pkl = OUT_DIR / 'hanseg_unet_pddca10_results_{recipe}.pkl'",
    )
    set_src(cell, s.rstrip("\n"))


def patch_heatmap_cell(cell, recipe: str):
    s = join_src(cell)
    titles = {
        "recipe1": "HaN-Seg U-Net (Reçete 1) → PDDCA 10 — DSC (%)",
        "recipe2": "HaN-Seg U-Net (Reçete 2) → PDDCA 10 — DSC (%)",
        "recipe3": "HaN-Seg U-Net (Reçete 3) → PDDCA 10 — DSC (%)",
    }
    s = s.replace(
        "ax.set_title('HaN-Seg U-Net → PDDCA 10 Test — DSC (%)')",
        f"ax.set_title('{titles[recipe]}')",
    )
    s = s.replace(
        "plt.savefig(OUT_DIR / 'hanseg_unet_heatmap.png', dpi=150, bbox_inches='tight')",
        f"plt.savefig(OUT_DIR / 'hanseg_unet_heatmap_{recipe}.png', dpi=150, bbox_inches='tight')",
    )
    set_src(cell, s.rstrip("\n"))


def patch_slice_dataset_r1(cell):
    s = join_src(cell)
    if "ndi.map_coordinates" in s:
        return
    insert_after = "            if np.random.rand() < 0.3:\n"
    idx = s.find(insert_after)
    if idx == -1:
        return
    # insert elastic after scale block - find "if np.random.rand() < 0.2:"Gaussian noise first block
    anchor = "            if np.random.rand() < 0.2:\n                img = img + np.random.normal"
    pos = s.find(anchor)
    if pos == -1:
        return
    s = s[:pos] + ELASTIC_BLOCK + "\n" + s[pos:]
    set_src(cell, s.rstrip("\n"))


RECIPES = [
    ("recipe1", "notebook-5_recipe1_nnunet_preprocessing.ipynb", MD_R1, CELL4_R1, "r1"),
    ("recipe2", "notebook-5_recipe2_soft_tissue_window.ipynb", MD_R2, CELL4_R2, "r2"),
    ("recipe3", "notebook-5_recipe3_small_organ_crop.ipynb", MD_R3, CELL4_R3, "r3"),
]


def build_nb(md, cell4, tag_key, recipe_short):
    nb = copy.deepcopy(nb_base)
    set_src(nb["cells"][0], md.strip())
    patch_config_cell(nb["cells"][2], recipe_short)
    set_src(nb["cells"][4], cell4.strip())
    patch_cache_cell(nb["cells"][6], tag_key, recipe_short)
    patch_test_cell(nb["cells"][8])
    patch_train_cell(nb["cells"][16], tag_key)
    patch_results_cell(nb["cells"][20], tag_key)
    patch_heatmap_cell(nb["cells"][22], tag_key)
    if recipe_short == "r1":
        patch_slice_dataset_r1(nb["cells"][10])
    return nb


for tag_key, out_name, md, c4, rshort in RECIPES:
    out_nb = build_nb(md, c4, tag_key, rshort)
    outp = BASE / out_name
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out_nb, f, ensure_ascii=False, indent=2)
    print("Wrote", outp)

print("Done.")
