"""
PDDCA — Ground Truth & Tahmin: Temiz 3D Görselleştirme
=======================================================
notebook-5_hanseg9organs_pddca10.ipynb ile BİREBİR AYNI
preprocessing ve model tanımı kullanır.

Kullanım — Notebook son hücresine:
    %run predict_and_visualize_3d.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# U-Net model tanımı — notebook-5 ile BİREBİR AYNI
# ═══════════════════════════════════════════════════════════════════════
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
            # ► Boyut uyumu — notebook-5'teki gibi ◄
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            # ► skip ÖNCE, x SONRA — notebook-5'teki gibi ◄
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.final_conv(x)


# ─── Sabitler ──────────────────────────────────────────────────────────
MODEL_CKPT = Path('hanseg_unet_best.pth')
NUM_CLASSES = 10
IMG_SIZE = 256
HU_MIN, HU_MAX = -200, 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════
# Preprocessing — notebook-5 ile BİREBİR AYNI (preprocess_volume)
# ═══════════════════════════════════════════════════════════════════════
def preprocess_volume(img_np, target_size=IMG_SIZE):
    """
    notebook-5'teki preprocess_volume fonksiyonunun aynısı.
    Tüm slice'ları HU normalizasyonu + resize yapar.
    Döndürür: (n_slices, target_size, target_size) float32 array
    """
    img = np.clip(img_np.astype(np.float32), HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    n_slices = img.shape[2]
    slices_img = []
    for s in range(n_slices):
        sl_img = img[:, :, s]
        if target_size != sl_img.shape[0]:
            zf = target_size / sl_img.shape[0]
            sl_img = ndi.zoom(sl_img, zf, order=1)
        slices_img.append(sl_img)
    return np.array(slices_img, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Tahmin — notebook-5'teki evaluate_patient ile BİREBİR AYNI
# ═══════════════════════════════════════════════════════════════════════
def predict_volume(model, preprocessed_slices, device):
    """
    notebook-5'teki evaluate_patient fonksiyonunun aynı tahmin mantığı.
    Girdi: (n_slices, H, W) preprocessed slices
    Çıktı: (n_slices, H, W) tahmin label'ları
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(preprocessed_slices)):
            # notebook-5: slices[i:i+1] → unsqueeze(0) → shape [1, 1, H, W]
            img_t = torch.from_numpy(preprocessed_slices[i:i+1]).unsqueeze(0).to(device)
            logits = model(img_t)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            preds.append(pred)
    return np.stack(preds, axis=0)


# ═══════════════════════════════════════════════════════════════════════
# Post-processing: Her organ maskesini ayrı ayrı temizle
# ═══════════════════════════════════════════════════════════════════════
def clean_prediction(pred_vol, num_classes=10, min_voxels=5):
    from scipy.ndimage import (label as nd_label, binary_opening,
                                binary_closing, binary_fill_holes,
                                generate_binary_structure, median_filter)

    # Küçük organlar — morfolojik işlem UYGULANMAZ (notebook-5 gibi raw argmax)
    # Chiasm=2, OpticNerve_L=4, OpticNerve_R=5
    SMALL_ORGANS = {2, 4, 5}

    struct_3d = generate_binary_structure(3, 2)
    cleaned = np.zeros_like(pred_vol)

    for c in range(1, num_classes):
        mask = (pred_vol == c).astype(np.uint8)
        if mask.sum() < min_voxels:
            continue

        if c in SMALL_ORGANS:
            # ── Küçük organ: sadece en büyük bileşeni tut, morfoloji YOK ──
            labeled_array, num_features = nd_label(mask, structure=struct_3d)
            if num_features == 0:
                continue
            component_sizes = np.bincount(labeled_array.ravel())
            component_sizes[0] = 0
            largest = component_sizes.argmax()
            best_mask = (labeled_array == largest)
            if best_mask.sum() >= min_voxels:
                cleaned[best_mask] = c
        else:
            # ── Büyük organ: tam post-processing ──
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
            best_mask = (labeled_array == largest)

            if best_mask.sum() >= min_voxels:
                cleaned[best_mask] = c

    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# ANA ÇALIŞMA BLOĞU
# ═══════════════════════════════════════════════════════════════════════

# 1) Modeli yükle
model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
state_dict = torch.load(MODEL_CKPT, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print("✓ Model yüklendi:", MODEL_CKPT)

# 2) Hasta seç ve veriyi yükle
PATIENT_ID = '0522c0708'  # Test hastası — iyi DSC skorları
img_vol, gt_label_vol = load_pddca_patient(PATIENT_ID)
print(f"✓ Hasta: {PATIENT_ID} | Volum: {img_vol.shape}")

# 3) Preprocessing — notebook-5 ile aynı
print("⏳ Preprocessing...")
preprocessed_slices = preprocess_volume(img_vol)
print(f"✓ Preprocessed: {preprocessed_slices.shape}")

# 4) Tahmin — notebook-5 ile aynı inference pipeline
print("⏳ Tahmin üretiliyor...")
pred_256 = predict_volume(model, preprocessed_slices, DEVICE)
print(f"✓ Tahmin (256x256): {pred_256.shape}")

# 5) Tahminleri orijinal boyuta geri ölçekle (3D görselleştirme için)
orig_h, orig_w = img_vol.shape[0], img_vol.shape[1]
if pred_256.shape[1] != orig_h or pred_256.shape[2] != orig_w:
    print(f"⏳ {pred_256.shape[1]}x{pred_256.shape[2]} → {orig_h}x{orig_w} resize...")
    pred_resized = []
    for s in range(pred_256.shape[0]):
        sl = ndi.zoom(pred_256[s], (orig_h / pred_256.shape[1], orig_w / pred_256.shape[2]), order=0)
        pred_resized.append(sl)
    pred_resized = np.stack(pred_resized, axis=0)
else:
    pred_resized = pred_256

# pred_resized shape: (n_slices, orig_h, orig_w) → transpose to (orig_h, orig_w, n_slices)
pred_label_vol_hwz = np.transpose(pred_resized, (1, 2, 0))
print(f"✓ Orijinal boyutta tahmin: {pred_label_vol_hwz.shape}")

# 6) Post-processing
print("⏳ Post-processing...")
pred_clean = clean_prediction(pred_label_vol_hwz, num_classes=NUM_CLASSES)

# Karşılaştırma tablosu
print("\nOrgan                | GT voxel    | Tahmin voxel")
print("-" * 55)
for c in range(1, NUM_CLASSES):
    gt_count = (gt_label_vol == c).sum()
    pred_count = (pred_clean == c).sum()
    name = STRUCTURE_NAMES[c - 1]
    print(f"  {name:20s} | {gt_count:>10,} | {pred_count:>10,}")

# 7) Ground Truth — temiz 3D
print("\n══ Ground Truth ══")
plot_organ_surfaces_3d_interactive(
    gt_label_vol,
    organs_to_show=list(range(1, 10)),
    step_size=2,
)

# 8) Tahmin — temizlenmiş, AYNI stil
print("\n══ Model Tahmini ══")
plot_organ_surfaces_3d_interactive(
    pred_clean,
    organs_to_show=list(range(1, 10)),
    step_size=2,
)
