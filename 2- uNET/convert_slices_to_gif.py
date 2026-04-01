#!/usr/bin/env python3
"""
ground_truth_slices/ altındaki her hasta klasöründeki slice_*.png dosyalarını
tek bir animasyonlu GIF'e birleştirir.
Çıktı: ground_truth_gifs/<hasta_id>.gif
"""
from pathlib import Path
from PIL import Image

SLICES_DIR = Path(__file__).resolve().parent / 'ground_truth_slices'
GIFS_DIR = Path(__file__).resolve().parent / 'ground_truth_gifs'
DURATION_MS = 150  # Her kare süresi (ms)


def main():
    GIFS_DIR.mkdir(exist_ok=True)

    if not SLICES_DIR.exists():
        print(f"Klasör bulunamadı: {SLICES_DIR}")
        return

    patient_dirs = sorted([d for d in SLICES_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')])

    for pdir in patient_dirs:
        slices = sorted(pdir.glob('slice_*.png'))
        if not slices:
            continue

        images = []
        for p in slices:
            img = Image.open(p).convert('RGB')
            images.append(img)

        out_path = GIFS_DIR / f'{pdir.name}.gif'
        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=DURATION_MS,
            loop=0,
        )
        print(f"{pdir.name}: {len(images)} kare → {out_path.name}")

    print(f"Tamamlandı: {GIFS_DIR.absolute()}")


if __name__ == '__main__':
    main()
