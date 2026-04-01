#!/usr/bin/env python3
"""
ground_truth_slices/ altındaki her hasta klasöründeki slice_*.png dosyalarını
MP4 video'ya dönüştürür (QuickTime ile açılır, sunum için ideal).
Çıktı: ground_truth_videos/<hasta_id>.mp4
"""
import subprocess
from pathlib import Path

SLICES_DIR = Path(__file__).resolve().parent / 'ground_truth_slices'
VIDEOS_DIR = Path(__file__).resolve().parent / 'ground_truth_videos'
FRAMERATE = 10
SCALE_WIDTH = 800


def main():
    VIDEOS_DIR.mkdir(exist_ok=True)

    if not SLICES_DIR.exists():
        print(f"Klasör bulunamadı: {SLICES_DIR}")
        return

    patient_dirs = sorted([d for d in SLICES_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')])

    for pdir in patient_dirs:
        slices = sorted(pdir.glob('slice_*.png'))
        if not slices:
            continue

        # ffmpeg -framerate 10 -i "slice_%03d.png" -vf "scale=800:-1" -c:v libx264 -pix_fmt yuv420p out.mp4
        out_path = VIDEOS_DIR / f'{pdir.name}.mp4'
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(FRAMERATE),
            '-i', str(pdir / 'slice_%03d.png'),
            '-vf', f'scale={SCALE_WIDTH}:-1',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"{pdir.name}: {len(slices)} kare → {out_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"[HATA] {pdir.name}: {e.stderr.decode() if e.stderr else e}")
        except FileNotFoundError:
            print("ffmpeg bulunamadı. 'brew install ffmpeg' ile yükleyin.")
            return

    print(f"Tamamlandı: {VIDEOS_DIR.absolute()}")


if __name__ == '__main__':
    main()
