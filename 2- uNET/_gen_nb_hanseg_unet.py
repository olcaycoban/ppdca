#!/usr/bin/env python3
"""Generate notebook: Train U-Net on HaN-Seg, test on PDDCA 10."""
import json
import textwrap
from pathlib import Path

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": textwrap.dedent(src).strip().splitlines(True)}

cells = []

cells.append(md("""
# U-Net — HaN-Seg (Sadece 9 Organ) Eğitim → PDDCA 10 Test

**Önemli:** HaN-Seg'de 30+ organ var; bu notebook **sadece PDDCA'daki 9 organ** ile eğitim yapar.

**Pipeline:**
1. **Eğitim:** HaN-Seg (42 hasta) — CT + **sadece 9 organ** (BrainStem, Chiasm, Mandible, OpticNerve_L/R, Parotid_L/R, Submandibular_L/R)
2. **Test:** PDDCA test_offsite (10 hasta)

**Kaggle:** İki dataset ekleyin → Add Input → PDDCA + HaN-Seg
- PDDCA: `data_split.json` içeren klasör
- HaN-Seg: `set_1/case_XX/` yapısı (Zenodo HaN-Seg veya benzeri)
- GPU: Açık olsun (Settings → Accelerator: GPU T4 x2)
"""))

cells.append(code(r"""
import os, importlib, subprocess, sys

def install_if_missing(package, import_name=None):
    import_name = import_name or package
    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

install_if_missing("pynrrd", "nrrd")
install_if_missing("openpyxl")

import json, time, pickle, warnings
from pathlib import Path
import numpy as np
import nrrd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
print("Datasets:", os.listdir('/kaggle/input'))
print(f"PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""))

cells.append(code(r"""
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    DEVICE = torch.device('cpu')

OUT_DIR = Path('/kaggle/working')

# Auto-detect PDDCA ve HaN-Seg
PDDCA_DIR = None
HANSEG_DIR = None
for d in sorted(os.listdir('/kaggle/input')):
    full = Path('/kaggle/input') / d
    if (full / 'data_split.json').exists():
        PDDCA_DIR = full
    # HaN-Seg: set_1 icinde case_XX klasorleri
    cand = full if (full / 'set_1').exists() else (full / 'HaN-Seg' if (full / 'HaN-Seg' / 'set_1').exists() else None)
    if cand is not None and (cand / 'set_1').exists():
        HANSEG_DIR = cand
    if HANSEG_DIR is None and ('hanseg' in d.lower() or 'han-seg' in d.lower()):
        for sub in [full, full / 'HaN-Seg']:
            if (sub / 'set_1').exists():
                HANSEG_DIR = sub
                break

assert PDDCA_DIR, "PDDCA (data_split.json) bulunamadi"
assert HANSEG_DIR, "HaN-Seg (set_1) bulunamadi"
print(f"PDDCA  : {PDDCA_DIR}")
print(f"HaN-Seg: {HANSEG_DIR}")

with open(PDDCA_DIR / 'data_split.json') as f:
    split = json.load(f)
TEST_IDS = split.get('test_offsite', split.get('test', []))

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

NUM_CLASSES = len(STRUCTURE_NAMES) + 1
IMG_SIZE = 256
HU_MIN, HU_MAX = -200, 300

BATCH_SIZE = 8
NUM_EPOCHS = 150
LR = 0.001
MOMENTUM = 0.99
WEIGHT_DECAY = 3e-4
GRAD_CLIP = 12.0
POLY_POWER = 0.9

print(f"Device: {DEVICE}  |  Test: {len(TEST_IDS)} hasta")
"""))

cells.append(md("## 1. HaN-Seg veri yükleme — Sadece PDDCA'daki 9 organ\n\n`HANSEG_OAR_MAP` ile HaN-Seg'deki karşılık gelen OAR dosyaları yüklenir. Diğer organlar kullanılmaz."))

cells.append(code(r"""
def load_hanseg_volume_and_masks(case_dir, case_id):
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


def preprocess_volume(img_np, label_np, target_size=IMG_SIZE):
    img = np.clip(img_np.astype(np.float32), HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    n_slices = img.shape[2]
    slices_img, slices_lbl = [], []
    for s in range(n_slices):
        sl_img = img[:, :, s]
        sl_lbl = label_np[:, :, s]
        if target_size != sl_img.shape[0]:
            zf = target_size / sl_img.shape[0]
            sl_img = ndi.zoom(sl_img, zf, order=1)
            sl_lbl = ndi.zoom(sl_lbl, zf, order=0)
        slices_img.append(sl_img)
        slices_lbl.append(sl_lbl)
    return np.array(slices_img, dtype=np.float32), np.array(slices_lbl, dtype=np.int64)

print("Yukleme fonksiyonlari hazir.")
"""))

cells.append(md("## 2. Eğitim verisi (HaN-Seg) önbelleğe al"))

cells.append(code(r"""
CACHE_PATH = OUT_DIR / 'hanseg_unet_train_cache.pkl'

if CACHE_PATH.exists():
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    train_slices = cache['train_slices']
    train_labels = cache['train_labels']
    print(f"Cache yuklendi: {len(train_slices)} train slice")
else:
    set_dir = HANSEG_DIR / 'set_1'
    cases = sorted([d for d in set_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])

    train_slices, train_labels = [], []
    print("HaN-Seg train yukleniyor...", flush=True)
    for i, case_dir in enumerate(cases):
        cid = case_dir.name
        try:
            img, lbl, _ = load_hanseg_volume_and_masks(case_dir, cid)
            sl_img, sl_lbl = preprocess_volume(img, lbl)
            for j in range(len(sl_img)):
                if sl_lbl[j].max() > 0:
                    train_slices.append(sl_img[j])
                    train_labels.append(sl_lbl[j])
                elif np.random.rand() < 0.2:
                    train_slices.append(sl_img[j])
                    train_labels.append(sl_lbl[j])
        except Exception as e:
            print(f"  [ATLA] {cid}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(cases)} hasta", flush=True)

    train_slices = np.array(train_slices)
    train_labels = np.array(train_labels)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'train_slices': train_slices, 'train_labels': train_labels}, f)
    print(f"Cache yazildi: {len(train_slices)} slice")

# Class weights
class_pixels = np.zeros(NUM_CLASSES, dtype=np.float64)
for c in range(NUM_CLASSES):
    class_pixels[c] = (train_labels == c).sum()
total = class_pixels.sum()
class_freq = class_pixels / total
median_freq = np.median(class_freq[class_freq > 0])
class_weights = np.ones(NUM_CLASSES, dtype=np.float32)
for c in range(NUM_CLASSES):
    if class_freq[c] > 0:
        class_weights[c] = median_freq / class_freq[c]
class_weights = np.clip(class_weights, 0.1, 10.0)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights hazir. Ornek: BG={class_weights[0]:.3f}, BrainStem={class_weights[1]:.3f}")
"""))

cells.append(md("## 3. Test verisi (PDDCA 10)"))

cells.append(code(r"""
test_data = {}
print("PDDCA test yukleniyor...")
for pid in TEST_IDS:
    pdir = PDDCA_DIR / pid
    if not (pdir / 'img.nrrd').exists():
        continue
    img, lbl, _ = load_pddca_volume_and_masks(pid)
    sl_img, sl_lbl = preprocess_volume(img, lbl)
    test_data[pid] = {'slices': sl_img, 'labels': sl_lbl}
print(f"Test: {len(test_data)} hasta")
"""))

cells.append(md("## 4. Dataset ve DataLoader"))

cells.append(code(r"""
class SliceDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images, self.labels = images, labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        lbl = self.labels[idx].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1).copy()
                lbl = np.flip(lbl, axis=1).copy()
            if np.random.rand() < 0.5:
                ang = np.random.uniform(-15, 15)
                img = ndi.rotate(img, ang, reshape=False, order=1, mode='nearest')
                lbl = ndi.rotate(lbl, ang, reshape=False, order=0, mode='nearest')
            if np.random.rand() < 0.3:
                sc = np.random.uniform(0.85, 1.15)
                img_s = ndi.zoom(img, sc, order=1)
                lbl_s = ndi.zoom(lbl, sc, order=0)
                h, w = img.shape
                hs, ws = img_s.shape
                if sc >= 1.0:
                    sh, sw = (hs - h) // 2, (ws - w) // 2
                    img, lbl = img_s[sh:sh+h, sw:sw+w], lbl_s[sh:sh+h, sw:sw+w]
                else:
                    img_new = np.zeros((h, w), dtype=img.dtype)
                    lbl_new = np.zeros((h, w), dtype=lbl.dtype)
                    ph, pw = (h - hs) // 2, (w - ws) // 2
                    img_new[ph:ph+hs, pw:pw+ws] = img_s
                    lbl_new[ph:ph+hs, pw:pw+ws] = lbl_s
                    img, lbl = img_new, lbl_new
            if np.random.rand() < 0.2:
                img = img + np.random.normal(0, 0.02, img.shape).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)
            if np.random.rand() < 0.2:
                img = np.clip(img * np.random.uniform(0.9, 1.1), 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(lbl.astype(np.int64))

train_ds = SliceDataset(train_slices, train_labels, augment=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
print(f"Train loader: {len(train_loader)} batch")
"""))

cells.append(md("## 5. U-Net modeli"))

cells.append(code(r"""
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
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
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.final_conv(x)

model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
print(f"UNet — {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params")
"""))

cells.append(md("## 6. Loss ve optimizer"))

cells.append(code(r"""
class DiceLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes, self.smooth = num_classes, smooth
        if class_weights is not None:
            self.register_buffer('cw', class_weights)
        else:
            self.cw = None

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        tgt = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (probs * tgt).sum(dim=(2, 3))
        card = probs.sum(dim=(2, 3)) + tgt.sum(dim=(2, 3))
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        dice = dice[:, 1:]
        gt_exists = tgt[:, 1:].sum(dim=(2, 3)) > 0
        w = self.cw[1:] if self.cw is not None else torch.ones(self.num_classes - 1, device=logits.device)
        w_mask = gt_exists.float() * w
        denom = w_mask.sum()
        if denom > 0:
            return 1.0 - (dice * w_mask).sum() / denom
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights, dice_w=0.5, ce_w=0.5):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes, class_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dw, self.cw = dice_w, ce_w
    def forward(self, logits, targets):
        return self.dw * self.dice_loss(logits, targets) + self.cw * self.ce_loss(logits, targets)

criterion = CombinedLoss(NUM_CLASSES, class_weights_t).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY)

def poly_lr(epoch, max_epochs, init_lr, power=POLY_POWER):
    return init_lr * (1.0 - epoch / max_epochs) ** power

def compute_dice_per_class(logits, targets, num_classes):
    preds = logits.argmax(dim=1)
    dices = []
    for c in range(1, num_classes):
        pred_c = (preds == c).float()
        true_c = (targets == c).float()
        inter, total = (pred_c * true_c).sum(), pred_c.sum() + true_c.sum()
        if total > 0:
            dices.append((2.0 * inter / total).item())
    return np.mean(dices) if dices else 0.0

print("Loss + optimizer hazir")
"""))

cells.append(md("## 7. Eğitim döngüsü"))

cells.append(code(r"""
CKPT_PATH = OUT_DIR / 'hanseg_unet_best.pth'
best_dice = 0.0
history = {'epoch': [], 'loss': [], 'dice': []}
t0 = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    lr = poly_lr(epoch - 1, NUM_EPOCHS, LR)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    model.train()
    run_loss, run_dice, n_b = 0.0, 0.0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        run_loss += loss.item()
        run_dice += compute_dice_per_class(logits.detach(), lbls, NUM_CLASSES)
        n_b += 1

    avg_loss = run_loss / n_b
    avg_dice = run_dice / n_b
    history['epoch'].append(epoch)
    history['loss'].append(avg_loss)
    history['dice'].append(avg_dice)

    if avg_dice > best_dice:
        best_dice = avg_dice
        torch.save(model.state_dict(), CKPT_PATH)

    if epoch % 15 == 0 or epoch <= 3:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  loss={avg_loss:.4f}  dice={avg_dice:.4f}", flush=True)

print(f"Egitim bitti: {(time.time()-t0)/60:.1f} dk  |  Best dice: {best_dice:.4f}")
"""))

cells.append(md("## 8. PDDCA 10 üzerinde test"))

cells.append(code(r"""
def evaluate_patient(model, slices, labels, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(slices)):
            img_t = torch.from_numpy(slices[i:i+1]).unsqueeze(0).to(device)
            logits = model(img_t)
            preds.append(logits.argmax(dim=1).squeeze(0).cpu().numpy())
    pred_vol = np.stack(preds, axis=0)
    organ_dice = {}
    for c_idx, name in enumerate(STRUCTURE_NAMES, start=1):
        p = (pred_vol == c_idx).astype(float)
        t = (labels == c_idx).astype(float)
        inter, total = (p * t).sum(), p.sum() + t.sum()
        organ_dice[name] = 1.0 if total == 0 and inter == 0 else (2.0 * inter / total if total > 0 else 0.0)
    return organ_dice

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

results = {}
for pid in test_data:
    d = evaluate_patient(model, test_data[pid]['slices'], test_data[pid]['labels'], DEVICE)
    results[pid] = d
    print(f"  {pid}  avg={np.mean(list(d.values()))*100:.2f}%")
"""))

cells.append(md("## 9. Sonuçlar ve Excel"))

cells.append(code(r"""
df_rows = []
for pid in results:
    for organ, d in results[pid].items():
        df_rows.append({'patient_id': pid, 'organ': organ, 'dice': d})
df = pd.DataFrame(df_rows)

avg_per_organ = {name: df[df['organ'] == name]['dice'].mean() * 100 for name in STRUCTURE_NAMES}
all_avg = np.mean(list(avg_per_organ.values()))

print("=" * 60)
print(f"{'Organ':<22} {'Dice (%)':>10}")
print("-" * 60)
for name in STRUCTURE_NAMES:
    print(f"{name:<22} {avg_per_organ[name]:>9.2f}")
print("-" * 60)
print(f"{'Ortalama':<22} {all_avg:>9.2f}")
print("=" * 60)

out_xlsx = OUT_DIR / 'hanseg_unet_pddca10_results.xlsx'
out_pkl = OUT_DIR / 'hanseg_unet_pddca10_results.pkl'

rows = [{'Patient': pid, **{o: round(results[pid][o] * 100, 2) for o in STRUCTURE_NAMES}} for pid in results]
rows.append({'Patient': 'AVERAGE', **{o: round(avg_per_organ[o], 2) for o in STRUCTURE_NAMES}})
pd.DataFrame(rows).to_excel(out_xlsx, index=False)

with open(out_pkl, 'wb') as f:
    pickle.dump({'results': results, 'avg_per_organ': avg_per_organ}, f)

print(f"\nKaydedildi: {out_xlsx}")
print(f"Kaydedildi: {out_pkl}")
"""))

cells.append(md("## 10. Görselleştirme"))

cells.append(code(r"""
matrix = np.array([[results[pid][o] * 100 for o in STRUCTURE_NAMES] for pid in results])
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn',
    xticklabels=STRUCTURE_NAMES, yticklabels=list(results.keys()),
    vmin=0, vmax=100, ax=ax)
ax.set_title('HaN-Seg U-Net → PDDCA 10 Test — DSC (%)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'hanseg_unet_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0", "file_extension": ".py", "mimetype": "text/x-python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = Path(__file__).resolve().parent / 'notebook-5_hanseg9organs_pddca10.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)
print(f"Notebook olusturuldu: {out}")
