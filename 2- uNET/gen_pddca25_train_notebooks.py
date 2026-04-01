#!/usr/bin/env python3
"""notebook-5_recipe*_... -> notebook-6_pddca25_train_recipe*_test_offsite10.ipynb"""
import json
from pathlib import Path

BASE = Path(__file__).parent
RECIPES = [
    (1, "notebook-5_recipe1_nnunet_preprocessing.ipynb", "notebook-6_pddca25_train_recipe1_test_offsite10.ipynb"),
    (2, "notebook-5_recipe2_soft_tissue_window.ipynb", "notebook-6_pddca25_train_recipe2_test_offsite10.ipynb"),
    (3, "notebook-5_recipe3_small_organ_crop.ipynb", "notebook-6_pddca25_train_recipe3_test_offsite10.ipynb"),
]

def markdown_intro(rnum: int) -> str:
    common = (
        "**HaN-Seg kullanılmaz.** Eğitim ve test tek kaynak: PDDCA NRRD (`img.nrrd` + `structures/`).\n\n"
        "**Split (`data_split.json`):**  \n"
        "- **Eğitim:** `train` → 25 hasta (`train_optional` dahil edilmez).  \n"
        "- **Test:** `test_offsite` → 10 hasta (`test_onsite` kullanılmaz).  \n\n"
        "**Kaggle:** Tek dataset input — PDDCA (`data_split.json` + hasta klasörleri).\n\n"
        f"**Çıktılar:** `pddca25_*_recipe{rnum}.*` (cache, checkpoint, Excel, heatmap).\n"
    )
    heads = {
        1: (
            "# U-Net — Reçete 1: yalnızca PDDCA (25 train → test_offsite 10)\n\n"
            "**Reçete 1 (nnU-Net tarzı):** ~1.5 mm izotropik örnekleme, vücut içi 0.5–99.5 "
            "persentil, Z-score; eğitimde elastik augment. Ayrıntı: `notebook-5_recipe1_...` "
            "içindeki `preprocess_volume`.\n\n"
        ),
        2: (
            "# U-Net — Reçete 2: yalnızca PDDCA (25 train → test_offsite 10)\n\n"
            "**Reçete 2 (yumuşak doku):** HU [-160, 240] → [0, 1] min–max. Ayrıntı: "
            "`notebook-5_recipe2_...` içindeki `preprocess_volume`.\n\n"
        ),
        3: (
            "# U-Net — Reçete 3: yalnızca PDDCA (25 train → test_offsite 10)\n\n"
            "**Reçete 3:** Kaba HU bbox ve küçük organ ağırlıklı örnekleme; ayrıntı: "
            "`notebook-5_recipe3_...` içindeki `preprocess_volume`.\n\n"
        ),
    }
    return heads[rnum] + common


def join_src(c):
    return "".join(c.get("source", []))


def set_src(c, text):
    lines = text.rstrip("\n").split("\n")
    c["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


def patch_config_cell(cell):
    s = join_src(cell)
    old = """OUT_DIR = Path('/kaggle/working')

# Auto-detect PDDCA ve HaN-Seg (rglob — Kaggle: Hansegv2 -> .../HaN-Seg/HaN-Seg/set_1)
IN_ROOT = Path('/kaggle/input')
PDDCA_DIR = None
for p in sorted(IN_ROOT.rglob('data_split.json')):
    PDDCA_DIR = p.parent
    break

def _find_hanseg_root():
    \"\"\"set_1 klasorunu bul; icinde case_* olan gercek train kokunu dondur.\"\"\"
    for set1 in sorted(IN_ROOT.rglob('set_1')):
        if not set1.is_dir():
            continue
        try:
            subs = list(set1.iterdir())
        except OSError:
            continue
        if any(d.is_dir() and d.name.startswith('case_') for d in subs):
            return set1.parent
    for p in IN_ROOT.rglob('*'):
        if not p.is_dir() or p.name.lower() != 'set_1':
            continue
        try:
            if any(d.is_dir() and d.name.startswith('case_') for d in p.iterdir()):
                return p.parent
        except OSError:
            continue
    return None

HANSEG_DIR = _find_hanseg_root()

assert PDDCA_DIR, \"PDDCA (data_split.json) bulunamadi\"
assert HANSEG_DIR, \"HaN-Seg (set_1) bulunamadi\"
print(f\"PDDCA  : {PDDCA_DIR}\")
print(f\"HaN-Seg: {HANSEG_DIR}\")

with open(PDDCA_DIR / 'data_split.json') as f:
    split = json.load(f)
TEST_IDS = split.get('test_offsite', split.get('test', []))"""

    new = """OUT_DIR = Path('/kaggle/working')

IN_ROOT = Path('/kaggle/input')
PDDCA_DIR = None
for p in sorted(IN_ROOT.rglob('data_split.json')):
    PDDCA_DIR = p.parent
    break

assert PDDCA_DIR, \"PDDCA (data_split.json) bulunamadi\"

with open(PDDCA_DIR / 'data_split.json') as f:
    split = json.load(f)

TRAIN_IDS = list(split['train'])
TEST_IDS = split.get('test_offsite', split.get('test', []))

print(f\"PDDCA  : {PDDCA_DIR}\")
print(f\"Train  : {len(TRAIN_IDS)} hasta (split['train'] — optional haric)\")
print(f\"Test   : {len(TEST_IDS)} hasta (test_offsite)\")"""

    if old not in s:
        raise ValueError("Config block not found — base recipe notebook degismis olabilir.")
    s = s.replace(old, new, 1)
    s = s.replace(
        "print(f\"Device: {DEVICE}  |  Test: {len(TEST_IDS)} hasta\")\nprint(f\"Train organ sayisi: {len(STRUCTURE_NAMES)} (sadece PDDCA'daki 9 organ)\")",
        "print(f\"Device: {DEVICE}  |  PDDCA 25 train / {len(TEST_IDS)} test\")\nprint(f\"Organ sinifi: {len(STRUCTURE_NAMES)} yapi + arka plan\")",
    )
    set_src(cell, s)


def remove_hanseg_map(cell):
    s = join_src(cell)
    i0 = s.find("HANSEG_OAR_MAP = {")
    if i0 == -1:
        set_src(cell, s)
        return
    i1 = s.find("}", i0)
    i1 = s.find("\n", i1) + 1
    s = s[:i0] + s[i1:]
    set_src(cell, s)


def remove_hanseg_loader(cell):
    s = join_src(cell)
    start = s.find("def load_hanseg_volume_and_masks")
    end = s.find("def load_pddca_volume_and_masks")
    if start != -1 and end != -1:
        s = s[:start] + s[end:]
    set_src(cell, s)


def patch_cache(cell, tag):
    s = join_src(cell)
    s = s.replace(
        f"CACHE_PATH = OUT_DIR / 'hanseg_unet_train_cache_{tag}.pkl'",
        f"CACHE_PATH = OUT_DIR / 'pddca25_train_cache_{tag}.pkl'",
    )
    old = """else:
    set_dir = HANSEG_DIR / 'set_1'
    cases = sorted([d for d in set_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])

    train_slices, train_labels = [], []
    print(\"HaN-Seg train yukleniyor...\", flush=True)
    for i, case_dir in enumerate(cases):
        cid = case_dir.name
        try:
            img, lbl, hdr = load_hanseg_volume_and_masks(case_dir, cid)
            sl_img, sl_lbl = preprocess_volume(img, lbl, hdr)"""

    new = """else:
    train_slices, train_labels = [], []
    print(\"PDDCA train yukleniyor (25 hasta)...\", flush=True)
    for i, pid in enumerate(TRAIN_IDS):
        pdir = PDDCA_DIR / pid
        if not (pdir / 'img.nrrd').exists():
            print(f\"  [ATLA] {pid}: img.nrrd yok\", flush=True)
            continue
        try:
            img, lbl, hdr = load_pddca_volume_and_masks(pid)
            sl_img, sl_lbl = preprocess_volume(img, lbl, hdr)"""

    if old not in s:
        raise ValueError("Cache HaN-Seg block not found")
    s = s.replace(old, new, 1)
    s = s.replace(
        "        except Exception as e:\n            print(f\"  [ATLA] {cid}: {e}\", flush=True)\n        if (i + 1) % 10 == 0:\n            print(f\"  {i+1}/{len(cases)} hasta\", flush=True)",
        "        except Exception as e:\n            print(f\"  [ATLA] {pid}: {e}\", flush=True)\n        if (i + 1) % 5 == 0:\n            print(f\"  {i+1}/{len(TRAIN_IDS)} hasta\", flush=True)",
    )
    set_src(cell, s)


def patch_train(cell, tag):
    s = join_src(cell)
    s = s.replace(
        f"CKPT_PATH = OUT_DIR / 'hanseg_unet_best_{tag}.pth'",
        f"CKPT_PATH = OUT_DIR / 'pddca25_unet_best_{tag}.pth'",
    )
    set_src(cell, s)


def patch_results(cell, tag):
    s = join_src(cell)
    s = s.replace(
        f"out_xlsx = OUT_DIR / 'hanseg_unet_pddca10_results_{tag}.xlsx'",
        f"out_xlsx = OUT_DIR / 'pddca25_train_offsite10_results_{tag}.xlsx'",
    )
    s = s.replace(
        f"out_pkl = OUT_DIR / 'hanseg_unet_pddca10_results_{tag}.pkl'",
        f"out_pkl = OUT_DIR / 'pddca25_train_offsite10_results_{tag}.pkl'",
    )
    set_src(cell, s)


def patch_heatmap(cell, tag, rnum):
    s = join_src(cell)
    s = s.replace(
        f"ax.set_title('HaN-Seg U-Net (Reçete {rnum}) → PDDCA 10 — DSC (%)')",
        f"ax.set_title('PDDCA 25 train (Reçete {rnum}) → test_offsite 10 — DSC (%)')",
    )
    s = s.replace(
        f"plt.savefig(OUT_DIR / 'hanseg_unet_heatmap_{tag}.png'",
        f"plt.savefig(OUT_DIR / 'pddca25_heatmap_{tag}.png'",
    )
    set_src(cell, s)


def main():
    for rnum, src_name, out_name in RECIPES:
        with open(BASE / src_name, "r", encoding="utf-8") as f:
            nb = json.load(f)
        set_src(nb["cells"][0], markdown_intro(rnum))
        patch_config_cell(nb["cells"][2])
        remove_hanseg_map(nb["cells"][2])
        remove_hanseg_loader(nb["cells"][4])
        tag = f"recipe{rnum}"
        patch_cache(nb["cells"][6], tag)
        nb["cells"][3]["source"] = [
            f"## 1. PDDCA — `load_pddca_volume_and_masks` + Reçete {rnum} preprocess\n",
            "\n",
            "Sadece PDDCA; harici veri seti yok.\n",
        ]
        nb["cells"][5]["source"] = [
            "## 2. Eğitim önbelleği — `split['train']` (25 hasta)\n",
            "\n",
            "`train_optional` eklenmez.\n",
        ]
        nb["cells"][7]["source"] = [
            "## 3. Test — `test_offsite` (10 hasta)\n",
            "\n",
            "`test_onsite` kullanılmaz.\n",
        ]
        patch_train(nb["cells"][16], tag)
        patch_results(nb["cells"][20], tag)
        patch_heatmap(nb["cells"][22], tag, rnum)
        c1 = nb["cells"][1]["source"]
        nb["cells"][1]["source"] = [
            (ln.replace(
                'print("Datasets:", os.listdir(\'/kaggle/input\'))',
                'print("Datasets:", os.listdir(\'/kaggle/input\') if os.path.isdir(\'/kaggle/input\') else \'(yerel)\')'
            ) if isinstance(ln, str) else ln)
            for ln in c1
        ]
        with open(BASE / out_name, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
        print("Wrote", out_name)


if __name__ == "__main__":
    main()
