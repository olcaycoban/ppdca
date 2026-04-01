#!/usr/bin/env python3
"""
Volume-level Dice değerlendirmesini açıklayan PDF oluşturur.
Çalıştırma: python generate_dice_explanation_pdf.py
"""
from pathlib import Path
import numpy as np

# Matplotlib ile PDF oluştur (tek sayfa veya çok sayfa)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

OUT_PDF = Path(__file__).resolve().parent / 'volume_dice_aciklamasi.pdf'

def main():
    with PdfPages(OUT_PDF) as pdf:
        # ========== SAYFA 1: Genel açıklama ==========
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.1, 0.95, 'Volume-Level Dice Değerlendirmesi', fontsize=18, fontweight='bold')
        fig.text(0.1, 0.90, 'HaN-Seg U-Net → PDDCA Test (notebook-4)', fontsize=12, style='italic')
        fig.text(0.1, 0.85, '─' * 80, fontsize=8)

        body = """
1. ÖN BİLGİ
   • Model 2D U-Net: Her CT slice'ı ayrı ayrı işler, slice bazında tahmin üretir.
   • Test verisi: PDDCA test hastaları (örn. 10 hasta).
   • Ground truth: Uzman tarafından elle çizilmiş organ maskeleri (structures/*.nrrd).

2. pred_vol NASIL ELDE EDİLİR?
   Her hasta için:
   • Slice 0: CT[:,:,0] → Model → pred_0 (256×256, sınıf etiketleri 0–9)
   • Slice 1: CT[:,:,1] → Model → pred_1
   • ...
   • Slice N-1: CT[:,:,N-1] → Model → pred_(N-1)

   pred_vol = np.stack([pred_0, pred_1, ..., pred_(N-1)], axis=0)
   → Şekil: (N_slice, 256, 256) — tüm volume'un tahmini

3. ORGAN BAZLI MASKE TANIMLARI
   Her organ c (c=1..9: BrainStem, Chiasm, Mandible, ...) için:

   p = tahmin maskesi: pred_vol == c olan voxel'ler (modelin c organını tahmin ettiği yerler)
   t = ground truth maskesi: labels == c olan voxel'ler (uzmanın işaretlediği yerler)

   Her ikisi de binary (0/1) maskelerdir ve tüm 3D volume üzerinde tanımlıdır.

4. DICE FORMÜLÜ
   Her organ c için:
                   2 × |p ∩ t|
   Dice(c) = ─────────────────────
                |p| + |t|

   • |p ∩ t| = p ve t'nin kesişimindeki voxel sayısı (doğru tahmin)
   • |p| = modelin tahmin ettiği voxel sayısı
   • |t| = ground truth'taki voxel sayısı

   Dice = 1 → Mükemmel uyum
   Dice = 0 → Hiç örtüşme yok
"""
        fig.text(0.1, 0.82, body, fontsize=10, verticalalignment='top', fontfamily='monospace',
                 wrap=True)

        plt.tight_layout(rect=[0, 0, 1, 0.82])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # ========== SAYFA 2: Görsel şema ==========
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Sol üst: Volume yapısı
        ax = axes[0, 0]
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('CT Volume → Slice bazlı tahmin', fontsize=11)
        # Basit kutu çizimi
        rect = mpatches.FancyBboxPatch((0.5, 1), 3, 2, boxstyle="round,pad=0.05", 
                                       facecolor='lightblue', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(2, 2, 'CT Volume\n(H×W×Z)', ha='center', va='center', fontsize=10)
        ax.annotate('', xy=(3.2, 2), xytext=(3.6, 2), arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(3.9, 2.2, 'Model', fontsize=9)
        rect2 = mpatches.FancyBboxPatch((4.5, 0.5), 2.5, 3, boxstyle="round,pad=0.05",
                                        facecolor='lightgreen', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect2)
        ax.text(5.75, 2, 'pred_vol\n(N×256×256)', ha='center', va='center', fontsize=9)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 4)

        # Sağ üst: Organ maskesi
        ax = axes[0, 1]
        ax.set_title('Organ c için p ve t maskeleri', fontsize=11)
        ax.axis('off')
        # 2D örnek gösterimi
        np.random.seed(42)
        t = np.zeros((8, 8))
        t[2:4, 2:5] = 1
        t[4:6, 3:6] = 1
        p = np.zeros((8, 8))
        p[2:5, 2:5] = 1
        p[5:7, 4:6] = 1
        overlap = p * t
        display = np.zeros((8, 8, 3))
        display[p.astype(bool), 0] = 1
        display[t.astype(bool), 1] = 1
        display[overlap.astype(bool), 2] = 1
        ax.imshow(display)
        ax.text(4, -0.8, 'Yeşil=t (GT)  Kırmızı=p (tahmin)  Mavi=kesişim', ha='center', fontsize=9)
        ax.text(4, 8.5, '|p∩t| = mavi piksel sayısı', ha='center', fontsize=9)

        # Alt: Örnek hesaplama
        ax = axes[1, 0]
        ax.axis('off')
        ax.set_title('Örnek hesaplama (Parotid_L, hasta 0522c0555)', fontsize=11)
        example_text = """
Örnek (sayısal):
  Organ: Parotid_L (Turuncu)
  Volume: 263 slice × 256 × 256

  Model tahmini (p):  12.450 voxel pozitif
  Ground truth (t):   11.200 voxel pozitif
  Kesişim (p ∩ t):    9.800 voxel

  Dice = 2 × 9800 / (12450 + 11200)
       = 19600 / 23650
       ≈ 0,83  (%83)
"""
        ax.text(0.05, 0.95, example_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        # Sonuç tablosu
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('Sonuç çıktısı (hasta × organ matrisi)', fontsize=11)
        orgs = ['BrainStem', 'Parotid_L', 'Mandible', '...']
        pats = ['0522c0555', '0522c0576', '0522c0598', '...']
        data = [[85.2, 78.1, 91.0, '...'],
                [72.3, 81.5, 88.2, '...'],
                [90.1, 75.0, 82.4, '...'],
                ['...', '...', '...', '...']]
        table = ax.table(cellText=data, rowLabels=pats, colLabels=orgs, loc='center',
                         cellLoc='center', colColours=['#e0e0e0']*4)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        ax.text(0.5, -0.05, 'Her hücre: O organının o hastadaki Dice (%)', ha='center',
                transform=ax.transAxes, fontsize=9)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # ========== SAYFA 3: Özet ==========
        fig = plt.figure(figsize=(8.5, 6))
        fig.text(0.1, 0.9, 'Özet', fontsize=16, fontweight='bold')
        fig.text(0.1, 0.85, '─' * 80, fontsize=8)
        summary = """
• Slice bazlı tahmin, volume bazlı değerlendirme:
  Model her slice'ı ayrı işler → pred_vol birleştirilir → Dice tüm volume üzerinden hesaplanır.

• Dice neden volume bazında?
  Tek bir organ (örn. Parotid_L) birden fazla slice'ta görünür. Doğruluk, organın tüm
  hacminin ne kadar iyi segment edildiğiyle ölçülür — tek bir slice yeterli değildir.

• Ground truth kullanımı:
  PDDCA structures/*.nrrd dosyaları → labels (H×W×Z, değerler 0–9).
  labels == c → Organ c'nin referans maskesi (uzman anotasyonu).

• Başarı oranı raporlama:
  — Hasta bazlı: Her hasta için 9 organın ortalama Dice'ı
  — Organ bazlı: Tüm hastalar üzerinde o organın ortalama Dice'ı
  — Genel: Tüm hasta–organ kombinasyonlarının ortalaması
"""
        fig.text(0.1, 0.82, summary, fontsize=11, verticalalignment='top', fontfamily='monospace',
                 wrap=True)
        plt.tight_layout(rect=[0, 0, 1, 0.82])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"PDF olusturuldu: {OUT_PDF.absolute()}")


if __name__ == '__main__':
    main()
