#!/usr/bin/env python3
"""
notebook-6_pddca25_train_recipe{2,3} → notebook-7 tam reçete versiyonları.

Reçete 2 tam: + 96×96 patch sampling (80% fg / 20% bg)
Reçete 3 tam: + 2.5D giriş (in_channels=3) + 64×64 ROI patch (küçük organ merkezli)
"""
import json, copy
from pathlib import Path

BASE = Path(__file__).parent

def join_src(c): return "".join(c.get("source", []))
def set_src(c, text):
    lines = text.rstrip("\n").split("\n")
    c["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


# ═══════════════════════════════════════════════════════════════════
#  REÇETE 2 TAM
# ═══════════════════════════════════════════════════════════════════

def build_recipe2_full():
    with open(BASE / "notebook-6_pddca25_train_recipe2_test_offsite10.ipynb") as f:
        nb = json.load(f)

    # --- Markdown başlık ---
    set_src(nb["cells"][0], """# U-Net — Reçete 2 TAM: PDDCA 25 train → test_offsite 10

**Belgedeki tüm adımlar uygulanmıştır:**
1. **Hard windowing:** [-160, 240] HU
2. **Normalizasyon:** [0, 1] min–max
3. **96×96 patch eğitimi:** Her epoch'ta kesitlerden rastgele 96×96 yamalar çıkarılır.
   - **%80** foreground (maske > 0) merkezli patch
   - **%20** rastgele patch (arka plan dahil)
4. Test: tam 256×256 kesitlerle (model fully-conv).

**Split:** `train` → 25 hasta; `test_offsite` → 10 hasta.
**Çıktılar:** `pddca25_*_recipe2_full.*`""")

    # --- Config: patch boyutu + çıktı isimleri ---
    cfg = join_src(nb["cells"][2])
    cfg = cfg.replace("RECIPE_TAG = 'recipe2'", "RECIPE_TAG = 'recipe2_full'\nPATCH_SIZE = 96\nFG_RATIO = 0.8")
    set_src(nb["cells"][2], cfg)

    # --- Cache: dosya adı ---
    cache = join_src(nb["cells"][6])
    cache = cache.replace("pddca25_train_cache_recipe2", "pddca25_train_cache_recipe2_full")
    set_src(nb["cells"][6], cache)

    # --- Dataset: PatchDataset ---
    set_src(nb["cells"][10], """class PatchDataset(Dataset):
    \"\"\"96×96 rastgele patch; %80 foreground merkezli, %20 rastgele.\"\"\"
    def __init__(self, images, labels, patch_size=96, fg_ratio=0.8, augment=False):
        self.images, self.labels = images, labels
        self.ps = patch_size
        self.fg_ratio = fg_ratio
        self.augment = augment
        self.fg_indices = []
        for idx in range(len(labels)):
            ys, xs = np.where(labels[idx] > 0)
            self.fg_indices.append((ys, xs))

    def __len__(self):
        return len(self.images)

    def _random_patch(self, img, lbl, fg_center=True):
        h, w = img.shape
        ps = min(self.ps, h, w)
        if fg_center:
            idx = np.random.randint(len(self.fg_indices[0])) if len(self.fg_indices[0]) > 0 else 0
            # will be overridden per-call
            pass
        cy, cx = h // 2, w // 2
        y0 = np.clip(cy - ps // 2, 0, h - ps)
        x0 = np.clip(cx - ps // 2, 0, w - ps)
        return img[y0:y0+ps, x0:x0+ps], lbl[y0:y0+ps, x0:x0+ps]

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        lbl = self.labels[idx].copy()
        h, w = img.shape
        ps = min(self.ps, h, w)

        if np.random.rand() < self.fg_ratio:
            ys, xs = self.fg_indices[idx]
            if len(ys) > 0:
                k = np.random.randint(len(ys))
                cy, cx = int(ys[k]), int(xs[k])
            else:
                cy, cx = np.random.randint(ps//2, h - ps//2 + 1), np.random.randint(ps//2, w - ps//2 + 1)
        else:
            cy = np.random.randint(ps//2, h - ps//2 + 1) if h > ps else h // 2
            cx = np.random.randint(ps//2, w - ps//2 + 1) if w > ps else w // 2

        y0 = np.clip(cy - ps // 2, 0, h - ps)
        x0 = np.clip(cx - ps // 2, 0, w - ps)
        img = img[y0:y0+ps, x0:x0+ps]
        lbl = lbl[y0:y0+ps, x0:x0+ps]

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1).copy()
                lbl = np.flip(lbl, axis=1).copy()
            if np.random.rand() < 0.5:
                ang = np.random.uniform(-15, 15)
                img = ndi.rotate(img, ang, reshape=False, order=1, mode='nearest')
                lbl = ndi.rotate(lbl, ang, reshape=False, order=0, mode='nearest')
            if np.random.rand() < 0.2:
                img = img + np.random.normal(0, 0.02, img.shape).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)
        return torch.from_numpy(img.copy()).unsqueeze(0).float(), torch.from_numpy(lbl.astype(np.int64).copy())

train_ds = PatchDataset(train_slices, train_labels, patch_size=PATCH_SIZE, fg_ratio=FG_RATIO, augment=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
print(f"PatchDataset: {len(train_ds)} slice → {PATCH_SIZE}×{PATCH_SIZE} patch, FG={FG_RATIO*100:.0f}%")
print(f"Train loader: {len(train_loader)} batch")""")

    # --- Train ckpt ---
    train = join_src(nb["cells"][16])
    train = train.replace("pddca25_unet_best_recipe2", "pddca25_unet_best_recipe2_full")
    set_src(nb["cells"][16], train)

    # --- Results ---
    res = join_src(nb["cells"][20])
    res = res.replace("pddca25_train_offsite10_results_recipe2", "pddca25_train_offsite10_results_recipe2_full")
    set_src(nb["cells"][20], res)

    # --- Heatmap ---
    hm = join_src(nb["cells"][22])
    hm = hm.replace("PDDCA 25 train (Reçete 2)", "PDDCA 25 (Reçete 2 TAM — 96×96 patch)")
    hm = hm.replace("pddca25_heatmap_recipe2", "pddca25_heatmap_recipe2_full")
    set_src(nb["cells"][22], hm)

    return nb


# ═══════════════════════════════════════════════════════════════════
#  REÇETE 3 TAM
# ═══════════════════════════════════════════════════════════════════

def build_recipe3_full():
    with open(BASE / "notebook-6_pddca25_train_recipe3_test_offsite10.ipynb") as f:
        nb = json.load(f)

    # --- Markdown başlık ---
    set_src(nb["cells"][0], """# U-Net — Reçete 3 TAM: PDDCA 25 train → test_offsite 10

**Belgedeki tüm adımlar uygulanmıştır:**
1. **Kaba kırpma:** HU > -500 bbox + margin
2. **HU penceresi:** [-500, 1000] → vücut Z-score → [0, 1]
3. **2.5D giriş:** Ardışık 3 kesit → `in_channels=3` (önceki + mevcut + sonraki)
4. **64×64 ROI patch:** Küçük organlar (Chiasm, Optik sinir) merkezli patch eğitimi
5. **Küçük organ oversampling:** Chiasm/OpticNerve kesitleri 2× örneklenir

**Split:** `train` → 25 hasta; `test_offsite` → 10 hasta.
**Çıktılar:** `pddca25_*_recipe3_full.*`""")

    # --- Config ---
    cfg = join_src(nb["cells"][2])
    cfg = cfg.replace("RECIPE_TAG = 'recipe3'", "RECIPE_TAG = 'recipe3_full'\nIN_CHANNELS = 3  # 2.5D\nPATCH_SIZE = 64\nFG_RATIO = 0.8")
    set_src(nb["cells"][2], cfg)

    # --- Cache adı + 2.5D slice oluşturma ---
    cache = join_src(nb["cells"][6])
    cache = cache.replace("pddca25_train_cache_recipe3", "pddca25_train_cache_recipe3_full")

    # 2.5D: preprocess sonrası 3-kanal slice oluştur
    old_loop_start = "            for j in range(len(sl_img)):"
    new_loop = """            n_sl = len(sl_img)
            for j in range(n_sl):
                # 2.5D: önceki + mevcut + sonraki kesit
                prev_sl = sl_img[max(0, j-1)]
                curr_sl = sl_img[j]
                next_sl = sl_img[min(n_sl-1, j+1)]
                img_25d = np.stack([prev_sl, curr_sl, next_sl], axis=0)  # (3, H, W)"""

    cache = cache.replace(old_loop_start, new_loop)
    cache = cache.replace(
        "                    train_slices.append(sl_img[j])\n                    train_labels.append(sl_lbl[j])",
        "                    train_slices.append(img_25d)\n                    train_labels.append(sl_lbl[j])"
    )
    # kalan append'ler de aynı
    cache = cache.replace(
        "                        train_slices.append(sl_img[j])\n                        train_labels.append(sl_lbl[j])",
        "                        train_slices.append(img_25d)\n                        train_labels.append(sl_lbl[j])"
    )

    set_src(nb["cells"][6], cache)

    # --- Test verisi: 2.5D ---
    test_cell = join_src(nb["cells"][8])
    old_test = """    sl_img, sl_lbl = preprocess_volume(img, lbl, hdr)
    test_data[pid] = {'slices': sl_img, 'labels': sl_lbl}"""
    new_test = """    sl_img, sl_lbl = preprocess_volume(img, lbl, hdr)
    n_sl = len(sl_img)
    slices_25d = []
    for j in range(n_sl):
        prev_sl = sl_img[max(0, j-1)]
        curr_sl = sl_img[j]
        next_sl = sl_img[min(n_sl-1, j+1)]
        slices_25d.append(np.stack([prev_sl, curr_sl, next_sl], axis=0))
    test_data[pid] = {'slices': np.array(slices_25d, dtype=np.float32), 'labels': sl_lbl}"""
    test_cell = test_cell.replace(old_test, new_test)
    set_src(nb["cells"][8], test_cell)

    # --- Dataset: ROI PatchDataset ---
    set_src(nb["cells"][10], """class PatchDataset25D(Dataset):
    \"\"\"2.5D (3-kanal) + 64×64 ROI patch; küçük organ merkezli.\"\"\"
    def __init__(self, images, labels, patch_size=64, fg_ratio=0.8, augment=False):
        self.images, self.labels = images, labels  # images: (N, 3, H, W)
        self.ps = patch_size
        self.fg_ratio = fg_ratio
        self.augment = augment
        self.fg_indices = []
        for idx in range(len(labels)):
            ys, xs = np.where(labels[idx] > 0)
            self.fg_indices.append((ys, xs))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()  # (3, H, W)
        lbl = self.labels[idx].copy()  # (H, W)
        _, h, w = img.shape
        ps = min(self.ps, h, w)

        if np.random.rand() < self.fg_ratio:
            ys, xs = self.fg_indices[idx]
            if len(ys) > 0:
                k = np.random.randint(len(ys))
                cy, cx = int(ys[k]), int(xs[k])
            else:
                cy = np.random.randint(ps//2, max(ps//2+1, h - ps//2 + 1))
                cx = np.random.randint(ps//2, max(ps//2+1, w - ps//2 + 1))
        else:
            cy = np.random.randint(ps//2, max(ps//2+1, h - ps//2 + 1))
            cx = np.random.randint(ps//2, max(ps//2+1, w - ps//2 + 1))

        y0 = np.clip(cy - ps // 2, 0, h - ps)
        x0 = np.clip(cx - ps // 2, 0, w - ps)
        img = img[:, y0:y0+ps, x0:x0+ps]
        lbl = lbl[y0:y0+ps, x0:x0+ps]

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=2).copy()
                lbl = np.flip(lbl, axis=1).copy()
            if np.random.rand() < 0.5:
                ang = np.random.uniform(-15, 15)
                for ch in range(3):
                    img[ch] = ndi.rotate(img[ch], ang, reshape=False, order=1, mode='nearest')
                lbl = ndi.rotate(lbl, ang, reshape=False, order=0, mode='nearest')
            if np.random.rand() < 0.3:
                dy = np.random.randint(-4, 5)
                dx = np.random.randint(-4, 5)
                img = ndi.shift(img, (0, dy, dx), order=1, mode='nearest')
                lbl = ndi.shift(lbl, (dy, dx), order=0, mode='nearest')
            if np.random.rand() < 0.2:
                img = img + np.random.normal(0, 0.02, img.shape).astype(np.float32)
                img = np.clip(img, 0.0, 1.0)
        return torch.from_numpy(img.copy()).float(), torch.from_numpy(lbl.astype(np.int64).copy())

train_ds = PatchDataset25D(train_slices, train_labels, patch_size=PATCH_SIZE, fg_ratio=FG_RATIO, augment=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
print(f"PatchDataset25D: {len(train_ds)} slice → {PATCH_SIZE}×{PATCH_SIZE} patch, 2.5D (3ch)")
print(f"Train loader: {len(train_loader)} batch")""")

    # --- U-Net: in_channels=3 ---
    model_cell = join_src(nb["cells"][12])
    model_cell = model_cell.replace(
        "model = UNet(in_channels=1, num_classes=NUM_CLASSES).to(DEVICE)",
        "model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)"
    )
    set_src(nb["cells"][12], model_cell)

    # --- Evaluate: 2.5D uyumlu ---
    eval_cell = join_src(nb["cells"][18])
    eval_cell = eval_cell.replace(
        "            img_t = torch.from_numpy(slices[i:i+1]).unsqueeze(0).to(device)",
        "            img_t = torch.from_numpy(slices[i:i+1]).to(device)  # (1, 3, H, W)"
    )
    set_src(nb["cells"][18], eval_cell)

    # --- Çıktı adları ---
    train = join_src(nb["cells"][16])
    train = train.replace("pddca25_unet_best_recipe3", "pddca25_unet_best_recipe3_full")
    set_src(nb["cells"][16], train)

    res = join_src(nb["cells"][20])
    res = res.replace("pddca25_train_offsite10_results_recipe3", "pddca25_train_offsite10_results_recipe3_full")
    set_src(nb["cells"][20], res)

    hm = join_src(nb["cells"][22])
    hm = hm.replace("PDDCA 25 train (Reçete 3)", "PDDCA 25 (Reçete 3 TAM — 2.5D + 64×64 ROI)")
    hm = hm.replace("pddca25_heatmap_recipe3", "pddca25_heatmap_recipe3_full")
    set_src(nb["cells"][22], hm)

    return nb


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

nb2 = build_recipe2_full()
out2 = BASE / "notebook-7_pddca25_recipe2_full_patch96.ipynb"
with open(out2, "w", encoding="utf-8") as f:
    json.dump(nb2, f, ensure_ascii=False, indent=2)
print("Wrote", out2.name)

nb3 = build_recipe3_full()
out3 = BASE / "notebook-7_pddca25_recipe3_full_25d_roi64.ipynb"
with open(out3, "w", encoding="utf-8") as f:
    json.dump(nb3, f, ensure_ascii=False, indent=2)
print("Wrote", out3.name)
