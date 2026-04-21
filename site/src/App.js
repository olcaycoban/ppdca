import { useEffect, useMemo, useState } from 'react';
import './App.css';

const publicBase = process.env.PUBLIC_URL || '';
const SCENE3D_CACHE_BUST = 'v14';
const MANIFEST_CACHE_BUST = 'v18';
const ATLAS_MANIFEST_CACHE_BUST = 'v2';
const SCENE_ATLAS_CACHE_BUST = 'v2';
const ATLAS_DICE_CACHE_BUST = 'v2';
const ATLAS_PAIRWISE_CACHE_BUST = 'v1';
const ATLAS_META_CACHE_BUST = 'v1';

function assetUrl(relPath) {
  const p = relPath.startsWith('/') ? relPath : `/${relPath}`;
  return `${publicBase}${p}`;
}

function diceCellStyle(pct) {
  if (pct == null || Number.isNaN(pct)) {
    return { background: '#e2e8f0', color: '#64748b' };
  }
  const x = Math.max(0, Math.min(100, pct));
  const hue = (x / 100) * 120;
  return { background: `hsl(${hue}, 72%, 88%)`, color: '#0f172a', fontWeight: 600 };
}

function fmtInt(n) {
  if (n == null || Number.isNaN(n)) return '—';
  return Number(n).toLocaleString('tr-TR');
}

function rowDice(organMetrics, dicePercent, name) {
  if (organMetrics?.[name] && 'dicePercent' in organMetrics[name]) {
    return organMetrics[name].dicePercent;
  }
  return dicePercent?.[name];
}

/* ── Translations ────────────────────────────────────────── */
const TRANSLATIONS = {
  tr: {
    siteTitle: 'PDDCA — Baş-Boyun Organ Segmentasyonu',
    siteSub: '2.5D U-Net ve Atlas tabanlı yöntemlerin karşılaştırmalı değerlendirmesi',
    darkMode: 'Gece', lightMode: 'Gündüz',
    nav: {
      viewer: '3D Görselleştirme', zranges: 'Organ Z-Aralıkları',
      dataset: 'Veri Seti', studies: 'Çalışmalar', contact: 'İletişim',
    },
    sidebar: {
      patient: 'Hasta', views: 'Gösterilecek görünümler',
      ready: 'hazır', pending: 'bekliyor',
      note: 'Sahneler arka planda üretiliyor. Sayfayı yenileyince yeni tamamlananlar görünür.',
    },
    slice: {
      btn: '2D Kesit Görüntüleyici', sub: 'HU · Ground Truth · Tahmin',
      generating: 'Sahne üretiliyor…', notReady: '2D kesit sahnesi henüz üretilmedi.',
    },
    dice: {
      heading: 'Dice (DSC, %)',
      note: 'Organ başına: 200 × kesişim / (GT vokseller + tahmin vokseller). Satıra tıklayınca sağda detay.',
      organ: 'Organ', dsc: 'DSC (%)',
      hint: 'Soldan bir organ adına tıklayın; Dice ve vokseller sağda görünür.',
      noData: 'Veri yok.',
      bothZero: 'Bu vakada hem GT hem tahminde bu organ yok (0 vokseller).',
      dscLabel: 'Dice (DSC):',
      intro: 'Aşağıdaki sayılar, üstteki 3B sahnesinde bu organ için gördüğünüz dört panelle aynıdır.',
      gtLabel: 'Ground truth (3D)', gtTotal: 'GT toplamı:',
      predLabel: 'Model tahmini (3D)', predTotal: 'Tahmin toplamı:',
      fnLabel: "GT'de olup tahminle uyuşmayan (GT etiketi)", fpLabel: 'Tahminde olup GT ile uyuşmayan (tahmin etiketi)',
      tpDeriv: "Kesişim (TP), hem GT'den FN çıkarılarak hem tahminden FP çıkarılarak bulunur:",
      gtVox: 'GT vokseller', predVox: 'Tahmin vokseller', intersection: 'Kesişim (TP)',
      denomNote: 'Payda için:', roundNote: '(küçük fark sayım yuvarlaması olabilir)',
      dscFormula1: 'DSC% (GT + tahmin paydası):', dscFormula2: 'Aynı sonuç (TP + FN + FP paydası):',
      upgradeHint: "Bu tablo için manifest'e organMetrics gerekir: mevcut scene_3d.html dosyalarından fill_manifest_organ_metrics_from_scene3d.py çalıştırın.",
    },
    atlas: {
      heading: 'Atlas — üç yöntem (09 notebook ile uyumlu)',
      top5: 'En iyi 5 atlas', top5suffix: '(MV / STAPLE seçimi; df_def genel ortalama Dice):',
      atlasSel: 'Atlas seçimi (Affine+Deformable çiftleri)',
      atlasSelAll: '25 atlas — organ başına ortalama Dice',
      best5: '(en iyi 5)',
      fusionHint: 'Notebook: en iyi 5 atlas ile birleşik tahmin; tek satır / organ.',
      organ: 'Organ', dsc: 'DSC (%)',
    },
    footer: {
      cacheTitle: 'Atlas önbellek dosya adları',
      cacheIntro: "Dice özeti: export_atlas_pkl_3d.py. Atlas 3B sahnesi ancak hacim pickle'ı / scene_3d.html üretildiğinde görünür.",
    },
    sceneNotReady: 'Sahne henüz üretilmedi',
    sceneNotReadyNote: 'Arka planda üretiliyor veya export betiği çalıştırılmamış.',
    loading: 'Yükleniyor…', manifestError: 'manifest.json yüklenemedi:',
    contact: {
      title: 'İletişim',
      lead: 'Bu çalışma hakkında sorularınız için aşağıdaki kanallardan ulaşabilirsiniz.',
    },
    dataset: {
      title: 'Veri Seti: PDDCA',
      lead: 'Public Domain Database for Computational Anatomy (PDDCA), baş-boyun bölgesinin BT görüntülerini ve uzman segmentasyon etiketlerini içeren açık erişimli bir tıbbi görüntüleme veri setidir.',
      generalInfo: 'Genel Bilgiler', structures: 'Etiketlenen Yapılar (9 organ)',
      dataSplit: 'Veri Bölme (Bu Çalışma)', source: 'Kaynak',
      modalite: 'Modalite', modaliteVal: 'Bilgisayarlı Tomografi (BT)',
      totalPatient: 'Toplam hasta', totalPatientVal: '48 (eğitim + test)',
      testOffsite: 'Bu çalışmada test (off-site)', testOffsiteVal: '10 hasta',
      imageSize: 'Görüntü boyutu', imageSizeVal: '~512 × 512 × 100–200 kesit',
      voxelSpacing: 'Voksel aralığı', voxelSpacingVal: '~0.98 × 0.98 × 3.0 mm',
      license: 'Lisans', licenseVal: 'Kamuya açık (Public Domain)',
      organs: [
        ['Sağ parotis bezi', '#e74c3c'], ['Sol parotis bezi', '#e67e22'],
        ['Sağ submandibular bez', '#f1c40f'], ['Sol submandibular bez', '#2ecc71'],
        ['Sağ optik sinir', '#1abc9c'], ['Sol optik sinir', '#3498db'],
        ['Beyin sapı (brainstem)', '#9b59b6'], ['Mandibula (çene kemiği)', '#34495e'],
        ['Chiasma (optik kiyazma)', '#e91e63'],
      ],
      splitSet: 'Küme', splitCount: 'Hasta sayısı', splitUse: 'Kullanım',
      train: 'Eğitim (train)', trainCount: '25', trainUse: 'Model eğitimi',
      val: 'Doğrulama (val)', valCount: '5', valUse: 'Hiperparametre ayarı',
      testIn: 'Test (in-site)', testInCount: '8', testInUse: 'Ara değerlendirme',
      testOff: 'Test (off-site)', testOffCount: '10', testOffUse: 'Son değerlendirme — bu sitede gösterilen',
    },
    studies: {
      title: 'Yapılan Çalışmalar',
      lead: 'PDDCA veri seti üzerinde iki farklı otomatik segmentasyon yaklaşımı uygulanmış ve karşılaştırılmıştır.',
      atlasTitle: 'Atlas Tabanlı Segmentasyon',
      atlasDesc: 'En yüksek kaliteli 5 atlas (eğitim hastası), SimpleITK ile her test hastasına kayıt edilmiş; kayıtlı etiketler üç farklı birleştirme stratejisiyle füze edilmiştir.',
      regLib: 'Kayıt kütüphanesi', regLibVal: 'SimpleITK',
      regType: 'Kayıt türü', regTypeVal: 'Afin + Deformable (Demons)',
      numAtlas: 'Atlas sayısı', numAtlasVal: '5 (en yüksek ortalama Dice)',
      fusionMethods: 'Birleştirme yöntemleri',
      hqDef: 'HQ Deformable', hqDefDesc: 'En iyi tek atlas; doğrudan deformable kayıt çıktısı kullanılır.',
      hqMV: 'HQ Deformable + Majority Voting', hqMVDesc: '5 atlasın kayıtlı etiketleri voksel bazında çoğunluk oylamasıyla birleştirilir.',
      hqSTAPLE: 'HQ Deformable + STAPLE', hqSTAPLEDesc: "Simultaneous Truth and Performance Level Estimation; her atlasın güvenilirliğini olasılıksal olarak modelleyen EM tabanlı füzyon.",
      resultsTitle: 'Organ Bazında Sonuçlar (Ortalama DSC %)',
      resultsNote: 'Ortalama, mevcut test hastaları üzerinden hesaplanmıştır.',
      organ: 'Organ', best: 'en iyi',
      specialTitle: 'U-Net 2.5D — Chiasm + Optik Sinir (Uzmanlaşmış Model)',
      specialDesc: 'Yalnızca Chiasm, OpticNerve_L ve OpticNerve_R hedef alınarak eğitilmiş özel bir 2.5D U-Net modeli. Ham hacim Z ekseninde k∈[50,220] aralığına kırpılmış (gözlerin bulunduğu kesitler), 3 sınıf segmentasyonu yapılmıştır.',
      specialDetails: [
        ['Giriş', '2.5D — 3 ardışık kesit (in_channels=3)'],
        ['ROI patch', '128×128, batch 16, AMP'],
        ['Z-kırpma', 'k ∈ [50, 220] (ham hacim ekseni)'],
        ['Epoch', '300 · early stopping (val Dice)'],
        ['Inference', 'Sliding-window (stride 64) + TTA (yatay flip)'],
        ['Sınıflar', 'Chiasm, OpticNerve_L, OpticNerve_R'],
      ],
      metricTitle: 'Değerlendirme Metriği',
      metricDesc: 'Segmentasyon kalitesi Dice Similarity Coefficient (DSC) ile ölçülmüştür. DSC = 2 × |A ∩ B| / (|A| + |B|), burada A tahmini, B ise ground truth segmentasyonudur. Değer aralığı: 0 (örtüşme yok) — 1 (mükemmel örtüşme).',
      unetTitle: 'U-Net (Derin Öğrenme)',
      unetDesc: 'PDDCA (25 eğitim, 10 test) üzerinde beş farklı U-Net konfigürasyonu denenmiştir. Sonuçlar 10 test hastasının ortalamasıdır (DSC %).',
      unetResultsNote: '10 test hastası ortalaması. En iyi değer her satırda vurgulanmıştır.',
      arch: 'Mimari', archVal: '2.5D U-Net (25-slice input)',
      roi: 'ROI boyutu', roiVal: '64 × 64 voksel',
      loss: 'Kayıp fonksiyonu', lossVal: 'Dice + Binary Cross-Entropy',
      opt: 'Optimizer', optVal: 'Adam', fw: 'Framework', fwVal: 'PyTorch',
    },
    zrange: {
      title: 'Organ Z-Aralığı Analizi',
      lead: '{n} hasta × 9 organ. Her organın axial kesitler boyunca kapladığı Z aralığı, kesit sayısına göre normalize edilmiştir.',
      gantt: 'Gantt (Z aralıkları)', span: 'Span Dağılımı', matrix: 'Eksik Organ Matrisi',
      ganttNote: 'X ekseni: Z kesiti / toplam kesit sayısı (normalize). Her satır bir hasta.',
      spanNote: 'Kutu: Q1–Q3 arası. Çizgi: medyan. Geniş alan: min–max. X: kesit sayısı.',
      matrixNote: 'Renkli = organ mevcut. ✕ = etiketsiz hasta.',
      min: 'min', mean: 'ort', max: 'max',
    },
  },
  en: {
    siteTitle: 'PDDCA — Head-Neck Organ Segmentation',
    siteSub: 'Comparative evaluation of 2.5D U-Net and atlas-based methods',
    darkMode: 'Dark', lightMode: 'Light',
    nav: {
      viewer: '3D Visualization', zranges: 'Organ Z-Ranges',
      dataset: 'Dataset', studies: 'Studies', contact: 'Contact',
    },
    sidebar: {
      patient: 'Patient', views: 'Visible views',
      ready: 'ready', pending: 'pending',
      note: 'Scenes are generated in the background. Refresh to see newly completed ones.',
    },
    slice: {
      btn: '2D Slice Viewer', sub: 'HU · Ground Truth · Prediction',
      generating: 'Scene generating…', notReady: '2D slice scene not yet generated.',
    },
    dice: {
      heading: 'Dice (DSC, %)',
      note: 'Per organ: 200 × intersection / (GT voxels + pred voxels). Click a row to see details.',
      organ: 'Organ', dsc: 'DSC (%)',
      hint: 'Click an organ name on the left to see Dice and voxel counts on the right.',
      noData: 'No data.',
      bothZero: 'Neither GT nor prediction has this organ in this case (0 voxels).',
      dscLabel: 'Dice (DSC):',
      intro: 'The numbers below match the four panels of the 3D scene above for this organ.',
      gtLabel: 'Ground truth (3D)', gtTotal: 'GT total:',
      predLabel: 'Model prediction (3D)', predTotal: 'Prediction total:',
      fnLabel: 'In GT but not matching prediction (GT label)', fpLabel: 'In prediction but not matching GT (pred label)',
      tpDeriv: 'Intersection (TP) is derived by subtracting FN from GT and FP from prediction:',
      gtVox: 'GT voxels', predVox: 'Pred voxels', intersection: 'Intersection (TP)',
      denomNote: 'Denominator check:', roundNote: '(small difference may be rounding)',
      dscFormula1: 'DSC% (GT + pred denominator):', dscFormula2: 'Same result (TP + FN + FP denominator):',
      upgradeHint: 'This table requires organMetrics in manifest: run fill_manifest_organ_metrics_from_scene3d.py from existing scene_3d.html files (no model needed).',
    },
    atlas: {
      heading: 'Atlas — three methods (notebook 09 compatible)',
      top5: 'Top 5 atlases', top5suffix: '(MV / STAPLE selection; df_def overall mean Dice):',
      atlasSel: 'Atlas selection (Affine+Deformable pairs)',
      atlasSelAll: '25 atlases — mean Dice per organ',
      best5: '(top 5)',
      fusionHint: 'Notebook: fused prediction using top 5 atlases; single row / organ.',
      organ: 'Organ', dsc: 'DSC (%)',
    },
    footer: {
      cacheTitle: 'Atlas cache file names',
      cacheIntro: 'Dice summary: export_atlas_pkl_3d.py. Atlas 3D scene only visible once volume pickle / scene_3d.html is generated.',
    },
    sceneNotReady: 'Scene not yet generated',
    sceneNotReadyNote: 'Generating in background or export script has not been run.',
    loading: 'Loading…', manifestError: 'Failed to load manifest.json:',
    contact: {
      title: 'Contact',
      lead: 'Reach out via the channels below for questions about this work.',
    },
    dataset: {
      title: 'Dataset: PDDCA',
      lead: 'The Public Domain Database for Computational Anatomy (PDDCA) is an open-access medical imaging dataset containing CT scans of the head-and-neck region with expert segmentation labels.',
      generalInfo: 'General Information', structures: 'Labeled Structures (9 organs)',
      dataSplit: 'Data Split (This Study)', source: 'Source',
      modalite: 'Modality', modaliteVal: 'Computed Tomography (CT)',
      totalPatient: 'Total patients', totalPatientVal: '48 (train + test)',
      testOffsite: 'Test (off-site, this study)', testOffsiteVal: '10 patients',
      imageSize: 'Image size', imageSizeVal: '~512 × 512 × 100–200 slices',
      voxelSpacing: 'Voxel spacing', voxelSpacingVal: '~0.98 × 0.98 × 3.0 mm',
      license: 'License', licenseVal: 'Public Domain',
      organs: [
        ['Right parotid gland', '#e74c3c'], ['Left parotid gland', '#e67e22'],
        ['Right submandibular gland', '#f1c40f'], ['Left submandibular gland', '#2ecc71'],
        ['Right optic nerve', '#1abc9c'], ['Left optic nerve', '#3498db'],
        ['Brain stem', '#9b59b6'], ['Mandible', '#34495e'],
        ['Optic chiasm', '#e91e63'],
      ],
      splitSet: 'Set', splitCount: 'Patients', splitUse: 'Usage',
      train: 'Training', trainCount: '25', trainUse: 'Model training',
      val: 'Validation', valCount: '5', valUse: 'Hyperparameter tuning',
      testIn: 'Test (in-site)', testInCount: '8', testInUse: 'Intermediate evaluation',
      testOff: 'Test (off-site)', testOffCount: '10', testOffUse: 'Final evaluation — shown on this site',
    },
    studies: {
      title: 'Studies',
      lead: 'Two different automatic segmentation approaches have been applied and compared on the PDDCA dataset.',
      atlasTitle: 'Atlas-Based Segmentation',
      atlasDesc: 'The 5 highest-quality atlases (training patients) were registered to each test patient using SimpleITK; registered labels were fused using three different fusion strategies.',
      regLib: 'Registration library', regLibVal: 'SimpleITK',
      regType: 'Registration type', regTypeVal: 'Affine + Deformable (Demons)',
      numAtlas: 'Number of atlases', numAtlasVal: '5 (highest mean Dice)',
      fusionMethods: 'Fusion methods',
      hqDef: 'HQ Deformable', hqDefDesc: 'Best single atlas; deformable registration output used directly.',
      hqMV: 'HQ Deformable + Majority Voting', hqMVDesc: 'Registered labels from 5 atlases are fused per voxel by majority vote.',
      hqSTAPLE: 'HQ Deformable + STAPLE', hqSTAPLEDesc: "Simultaneous Truth and Performance Level Estimation; EM-based fusion that probabilistically models each atlas's reliability.",
      resultsTitle: 'Per-Organ Results (Mean DSC %)',
      resultsNote: 'Mean computed over available test patients.',
      organ: 'Organ', best: 'best',
      specialTitle: 'U-Net 2.5D — Chiasm + Optic Nerve (Specialized Model)',
      specialDesc: 'A specialized 2.5D U-Net trained exclusively on Chiasm, OpticNerve_L and OpticNerve_R. The raw volume was Z-cropped to k∈[50,220] slices (eye region), performing 3-class segmentation.',
      specialDetails: [
        ['Input', '2.5D — 3 consecutive slices (in_channels=3)'],
        ['ROI patch', '128×128, batch 16, AMP'],
        ['Z-crop', 'k ∈ [50, 220] (raw volume axis)'],
        ['Epochs', '300 · early stopping (val Dice)'],
        ['Inference', 'Sliding-window (stride 64) + TTA (horizontal flip)'],
        ['Classes', 'Chiasm, OpticNerve_L, OpticNerve_R'],
      ],
      metricTitle: 'Evaluation Metric',
      metricDesc: 'Segmentation quality is measured using the Dice Similarity Coefficient (DSC). DSC = 2 × |A ∩ B| / (|A| + |B|), where A is the prediction and B is the ground truth. Range: 0 (no overlap) — 1 (perfect overlap).',
      unetTitle: 'U-Net (Deep Learning)',
      unetDesc: 'Five different U-Net configurations were tested on PDDCA (25 train, 10 test). Results are averaged over 10 test patients (DSC %).',
      unetResultsNote: 'Mean over 10 test patients. Best value per row is highlighted.',
      arch: 'Architecture', archVal: '2.5D U-Net (25-slice input)',
      roi: 'ROI size', roiVal: '64 × 64 voxels',
      loss: 'Loss function', lossVal: 'Dice + Binary Cross-Entropy',
      opt: 'Optimizer', optVal: 'Adam', fw: 'Framework', fwVal: 'PyTorch',
    },
    zrange: {
      title: 'Organ Z-Range Analysis',
      lead: '{n} patients × 9 organs. Z-range occupied by each organ along axial slices, normalised by total slice count.',
      gantt: 'Gantt (Z ranges)', span: 'Span Distribution', matrix: 'Missing Organ Matrix',
      ganttNote: 'X axis: Z slice / total slice count (normalised). Each row is one patient.',
      spanNote: 'Box: Q1–Q3. Line: median. Wide area: min–max. X: slice count.',
      matrixNote: 'Coloured = organ present. ✕ = unlabelled in that patient.',
      min: 'min', mean: 'avg', max: 'max',
    },
  },
};

/* ── Dice components ─────────────────────────────────────── */

function DiceDetailPanel({ organName, organMetrics, dicePercent, t }) {
  const d = t.dice;
  if (!organName) {
    return (
      <div className="dice-detail dice-detail-placeholder">
        <p className="dice-detail-hint">{d.hint}</p>
      </div>
    );
  }

  const m = organMetrics?.[organName];
  const fallbackPct = dicePercent?.[organName];

  if (!m && fallbackPct == null) {
    return (
      <div className="dice-detail">
        <h3 className="dice-detail-title">{organName}</h3>
        <p>{d.noData}</p>
      </div>
    );
  }

  if (!m) {
    return (
      <div className="dice-detail">
        <h3 className="dice-detail-title">{organName}</h3>
        <p className="dice-detail-dsc">
          {d.dscLabel} <strong>{fallbackPct.toFixed(1)} %</strong>
        </p>
        <p className="dice-detail-upgrade">{d.upgradeHint}</p>
      </div>
    );
  }

  const { dicePercent: dsc, gtVoxels, predVoxels, tpVoxels, fnVoxels, fpVoxels } = m;
  const empty = (gtVoxels ?? 0) === 0 && (predVoxels ?? 0) === 0;

  if (empty) {
    return (
      <div className="dice-detail">
        <h3 className="dice-detail-title">{organName}</h3>
        <p>{d.bothZero}</p>
      </div>
    );
  }

  const gt = gtVoxels ?? 0, pred = predVoxels ?? 0;
  const tp = tpVoxels ?? 0, fn = fnVoxels ?? 0, fp = fpVoxels ?? 0;
  const denom = gt + pred, sumParts = 2 * tp + fn + fp;
  const formulaGtPred = denom > 0 ? `200 × ${fmtInt(tp)} / (${fmtInt(gt)} + ${fmtInt(pred)})` : '';
  const formulaTpFnFp = sumParts > 0 ? `100 × (2 × ${fmtInt(tp)}) / (2 × ${fmtInt(tp)} + ${fmtInt(fn)} + ${fmtInt(fp)})` : '';

  return (
    <div className="dice-detail">
      <h3 className="dice-detail-title">{organName}</h3>
      <p className="dice-detail-dsc">
        {d.dscLabel} <strong>{dsc != null ? `${dsc.toFixed(1)} %` : '—'}</strong>
      </p>
      <p className="dice-detail-intro">{d.intro}</p>
      <ul className="dice-detail-quadrants">
        <li><strong>{d.gtLabel}</strong> → {d.gtTotal} <strong>{fmtInt(gt)}</strong></li>
        <li><strong>{d.predLabel}</strong> → {d.predTotal} <strong>{fmtInt(pred)}</strong></li>
        <li><strong>{d.fnLabel}</strong> → FN: <strong>{fmtInt(fn)}</strong></li>
        <li><strong>{d.fpLabel}</strong> → FP: <strong>{fmtInt(fp)}</strong></li>
      </ul>
      <p className="dice-detail-tpderiv">
        <strong>{d.intersection}</strong>, {d.tpDeriv}{' '}
        <span className="dice-detail-mono">TP = GT − FN = {fmtInt(gt)} − {fmtInt(fn)} = {fmtInt(tp)}</span>
        {' · '}
        <span className="dice-detail-mono">TP = Pred − FP = {fmtInt(pred)} − {fmtInt(fp)} = {fmtInt(tp)}</span>
      </p>
      <dl className="dice-detail-dl dice-detail-dl-compact">
        <dt>{d.gtVox}</dt><dd>{fmtInt(gt)}</dd>
        <dt>{d.predVox}</dt><dd>{fmtInt(pred)}</dd>
        <dt>{d.intersection}</dt><dd>{fmtInt(tp)}</dd>
        <dt>FN</dt><dd>{fmtInt(fn)}</dd>
        <dt>FP</dt><dd>{fmtInt(fp)}</dd>
      </dl>
      <p className="dice-detail-identity">
        {d.denomNote} <span className="dice-detail-mono">GT + Pred = 2×TP + FN + FP</span> →{' '}
        <strong>{fmtInt(denom)}</strong> = <strong>{fmtInt(sumParts)}</strong>
        {denom !== sumParts && <span className="dice-detail-warn"> {d.roundNote}</span>}.
      </p>
      {formulaGtPred && dsc != null && (
        <div className="dice-detail-formulas">
          <p className="dice-detail-formula">
            <strong>{d.dscFormula1}</strong> {formulaGtPred} ≈ <strong>{dsc.toFixed(1)}</strong>
          </p>
          {formulaTpFnFp && (
            <p className="dice-detail-formula">
              <strong>{d.dscFormula2}</strong> {formulaTpFnFp} ≈ <strong>{dsc.toFixed(1)}</strong>
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function DicePanel({ patientId, structureNames, organMetrics, dicePercent, selectedOrgan, onSelectOrgan, t }) {
  const d = t.dice;
  const hasAny =
    (organMetrics && Object.keys(organMetrics).length > 0) ||
    (dicePercent && Object.keys(dicePercent).length > 0);
  if (!hasAny || !structureNames?.length) return null;

  return (
    <section className="dice-section dice-section-wide" aria-label="Dice scores">
      <h2 className="dice-heading">
        {t.sidebar.patient} <span className="dice-pid">{patientId}</span> · {d.heading}
      </h2>
      <p className="dice-note">{d.note}</p>
      <div className="dice-split">
        <div className="dice-table-wrap">
          <table className="dice-table">
            <thead>
              <tr>
                <th scope="col">{d.organ}</th>
                <th scope="col">{d.dsc}</th>
              </tr>
            </thead>
            <tbody>
              {structureNames.map((name) => {
                const v = rowDice(organMetrics, dicePercent, name);
                const sel = selectedOrgan === name;
                return (
                  <tr key={name} className={sel ? 'dice-row dice-row-selected' : 'dice-row'}>
                    <td className="dice-organ">
                      <button type="button" className="dice-organ-btn" onClick={() => onSelectOrgan(name)} aria-pressed={sel}>
                        {name}
                      </button>
                    </td>
                    <td className="dice-val" style={diceCellStyle(v)}>
                      {v != null ? v.toFixed(1) : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <DiceDetailPanel organName={selectedOrgan} organMetrics={organMetrics} dicePercent={dicePercent} t={t} />
      </div>
    </section>
  );
}

/* ── Atlas components ────────────────────────────────────── */

const ATLAS_CACHE_FILES = [
  { file: 'train25_test10_hq_deformable_cache.pkl', note: 'HQ deformable kayıt — önbellek' },
  { file: 'train25_test10_hq_majority_voting_deformable_cache.pkl', note: 'HQ deformable + çoğunluk oylaması — önbellek' },
  { file: 'train25_test10_hq_staple_deformable_cache.pkl', note: 'HQ deformable + STAPLE — önbellek' },
];

function AtlasDiceTriplet({ patientId, diceDoc, structureNames, pairwiseDoc, atlasMeta, t }) {
  const at = t.atlas;
  const methods = diceDoc?.methods;
  const [deformableAtlas, setDeformableAtlas] = useState('mean');

  useEffect(() => { setDeformableAtlas('mean'); }, [patientId]);

  if (!methods?.length) return null;

  const organs = structureNames || diceDoc.structureNames || [];
  const top5 = atlasMeta?.top5AtlasIdsForFusion || pairwiseDoc?.top5AtlasIdsByMeanDice || [];
  const atlasOrder = pairwiseDoc?.atlasIdsDisplayOrder || [];

  return (
    <section className="atlas-dice-triplet" aria-label="Atlas Dice three methods">
      <h2 className="atlas-dice-triplet-heading">{at.heading}</h2>
      <p className="atlas-dice-triplet-note">{diceDoc.note}</p>
      {top5.length > 0 && (
        <p className="atlas-dice-triplet-top5">
          <strong>{at.top5}</strong> {at.top5suffix}{' '}
          <span className="atlas-dice-top5-ids">{top5.join(', ')}</span>
        </p>
      )}
      <div className="atlas-dice-triplet-grid">
        {methods.map((m) => {
          let byOrg = m.diceByPatient?.[patientId] || {};
          if (m.id === 'deformable' && pairwiseDoc?.byTarget && deformableAtlas !== 'mean') {
            byOrg = pairwiseDoc.byTarget[patientId]?.[deformableAtlas] || {};
          }
          return (
            <div key={m.id} className="atlas-dice-triplet-col">
              <h3 className="atlas-dice-triplet-method">{m.label}</h3>
              <p className="atlas-dice-triplet-pkl">
                <code className="project-refs-mono">{m.pkl}</code>
              </p>
              {m.id === 'deformable' && atlasOrder.length > 0 ? (
                <div className="atlas-dice-atlas-select-wrap">
                  <label className="atlas-dice-atlas-label" htmlFor="atlas-deformable-select">
                    {at.atlasSel}
                  </label>
                  <select
                    id="atlas-deformable-select"
                    className="atlas-dice-atlas-select"
                    value={deformableAtlas}
                    onChange={(e) => setDeformableAtlas(e.target.value)}
                  >
                    <option value="mean">{at.atlasSelAll}</option>
                    {atlasOrder.map((aid) => (
                      <option key={aid} value={aid}>
                        {aid}{top5.includes(aid) ? ` ${at.best5}` : ''}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}
              {(m.id === 'majority_voting' || m.id === 'staple') && (
                <p className="atlas-dice-fusion-hint">{at.fusionHint}</p>
              )}
              <table className="atlas-dice-mini-table">
                <thead>
                  <tr><th scope="col">{at.organ}</th><th scope="col">{at.dsc}</th></tr>
                </thead>
                <tbody>
                  {organs.map((name) => {
                    const v = byOrg[name];
                    const pct = v != null ? v * 100 : null;
                    return (
                      <tr key={name}>
                        <td>{name}</td>
                        <td className="atlas-dice-mini-val" style={diceCellStyle(pct)}>
                          {pct != null ? pct.toFixed(1) : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/* ── Viewer item definitions ─────────────────────────────── */

const VIEWER_ITEMS = [
  { id: 'gt',             label: 'Ground Truth',            shortLabel: 'GT',           iframeHeight: 560 },
  { id: 'unet',           label: 'U-Net tahmini',           shortLabel: 'U-Net',        iframeHeight: 560 },
  { id: 'unet_fn',        label: 'U-Net FN',                shortLabel: 'U-Net FN',     iframeHeight: 560 },
  { id: 'unet_fp',        label: 'U-Net FP',                shortLabel: 'U-Net FP',     iframeHeight: 560 },
  { id: 'deformable',     label: 'HQ Deformable',           shortLabel: 'Deformable',   iframeHeight: 560 },
  { id: 'majority_voting',label: 'HQ Deformable + MV',      shortLabel: 'MV',           iframeHeight: 560 },
  { id: 'staple',         label: 'HQ Deformable + STAPLE',  shortLabel: 'STAPLE',       iframeHeight: 560 },
];

function AtlasArtifactsFooter({ t }) {
  const f = t.footer;
  return (
    <footer className="project-refs" aria-label="Atlas cache files">
      <h2 className="project-refs-title">{f.cacheTitle}</h2>
      <p className="project-refs-intro">{f.cacheIntro}</p>
      <ul className="project-refs-list">
        {ATLAS_CACHE_FILES.map(({ file, note }) => (
          <li key={file} className="project-refs-item">
            <code className="project-refs-mono">{file}</code>
            <span className="project-refs-dash"> — </span>
            <span>{note}</span>
          </li>
        ))}
      </ul>
    </footer>
  );
}

/* ── Z-Range page ────────────────────────────────────────── */

const ORGAN_ORDER = [
  'Mandible','BrainStem','Parotid_L','Parotid_R',
  'Submandibular_L','Submandibular_R','OpticNerve_L','OpticNerve_R','Chiasm',
];
const ORGAN_HEX = {
  BrainStem:'#ff3333', Chiasm:'#ffcc00', Mandible:'#33cc33',
  OpticNerve_L:'#0099ff', OpticNerve_R:'#9900ff',
  Parotid_L:'#ff8000', Parotid_R:'#ff0080',
  Submandibular_L:'#00cccc', Submandibular_R:'#cc9933',
};
const Z_CACHE = 'v1';

function ZRangePage({ t }) {
  const zt = t.zrange;
  const [data, setData] = useState(null);
  const [view, setView] = useState('gantt');
  const [hoveredOrgan, setHoveredOrgan] = useState(null);

  useEffect(() => {
    fetch(`${assetUrl('pddca-viz-atlas/z_ranges.json')}?cb=${Z_CACHE}`)
      .then(r => r.json()).then(setData).catch(() => setData([]));
  }, []);

  if (!data) return <div className="static-page"><p style={{color:'#94a3b8'}}>{t.loading}</p></div>;

  return (
    <div className="static-page zr-page">
      <h2 className="static-page-title">{zt.title}</h2>
      <p className="static-page-lead">{zt.lead.replace('{n}', data.length)}</p>

      <div className="zr-legend">
        {ORGAN_ORDER.map(o => (
          <span key={o}
            className={`zr-leg-item${hoveredOrgan === o ? ' zr-leg-item-on' : ''}`}
            onMouseEnter={() => setHoveredOrgan(o)}
            onMouseLeave={() => setHoveredOrgan(null)}>
            <span className="zr-leg-dot" style={{background: ORGAN_HEX[o]}}/>
            {o}
          </span>
        ))}
      </div>

      <div className="zr-tabs">
        {[['gantt', zt.gantt], ['span', zt.span], ['matrix', zt.matrix]].map(([id, label]) => (
          <button key={id} className={`zr-tab${view===id?' zr-tab-on':''}`} onClick={() => setView(id)}>{label}</button>
        ))}
      </div>

      {view === 'gantt'  && <ZRangeGantt  data={data} hoveredOrgan={hoveredOrgan} onHover={setHoveredOrgan} t={t}/>}
      {view === 'span'   && <ZRangeSpan   data={data} hoveredOrgan={hoveredOrgan} onHover={setHoveredOrgan} t={t}/>}
      {view === 'matrix' && <ZRangeMatrix data={data} t={t}/>}
    </div>
  );
}

function ZRangeGantt({ data, hoveredOrgan, onHover, t }) {
  const zt = t.zrange;
  return (
    <div className="zr-gantt-wrap">
      <div className="zr-gantt-axis">
        {[0,25,50,75,100].map(p => (
          <span key={p} className="zr-gantt-tick" style={{left:`${p}%`}}>{p}%</span>
        ))}
      </div>
      <div className="zr-gantt-rows">
        {data.map(patient => (
          <div key={patient.id} className="zr-gantt-row">
            <span className="zr-gantt-pid">{patient.id.replace('0522c','')}</span>
            <div className="zr-gantt-bar-area">
              {ORGAN_ORDER.map(organ => {
                const o = patient.organs[organ];
                if (!o?.present) return null;
                const left  = (o.z_min / patient.total_z * 100).toFixed(2);
                const width = ((o.z_max - o.z_min + 1) / patient.total_z * 100).toFixed(2);
                const faded = hoveredOrgan && hoveredOrgan !== organ;
                return (
                  <div key={organ}
                    title={`${organ}: z ${o.z_min}–${o.z_max} (${o.span})`}
                    className={`zr-bar${faded?' zr-bar-faded':''}`}
                    style={{left:`${left}%`, width:`${width}%`, background: ORGAN_HEX[organ]}}
                    onMouseEnter={() => onHover(organ)}
                    onMouseLeave={() => onHover(null)}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <p className="zr-note">{zt.ganttNote}</p>
    </div>
  );
}

function ZRangeSpan({ data, hoveredOrgan, onHover, t }) {
  const zt = t.zrange;
  const stats = useMemo(() => {
    return ORGAN_ORDER.map(organ => {
      const spans = data.map(p => p.organs[organ]).filter(o => o?.present).map(o => o.span);
      if (!spans.length) return { organ, min:0, max:0, median:0, mean:0, q1:0, q3:0 };
      spans.sort((a,b)=>a-b);
      const n = spans.length;
      const mean = spans.reduce((s,v)=>s+v,0)/n;
      const median = n%2===0 ? (spans[n/2-1]+spans[n/2])/2 : spans[Math.floor(n/2)];
      const q1 = spans[Math.floor(n*0.25)];
      const q3 = spans[Math.floor(n*0.75)];
      return { organ, min:spans[0], max:spans[n-1], mean:+mean.toFixed(1), median, q1, q3, spans };
    });
  }, [data]);

  const maxSpan = Math.max(...stats.map(s => s.max));

  return (
    <div className="zr-span-wrap">
      {stats.map(s => {
        const faded = hoveredOrgan && hoveredOrgan !== s.organ;
        const col = ORGAN_HEX[s.organ];
        const pct = v => (v / maxSpan * 100).toFixed(1);
        return (
          <div key={s.organ}
            className={`zr-span-row${faded?' zr-span-faded':''}`}
            onMouseEnter={() => onHover(s.organ)}
            onMouseLeave={() => onHover(null)}>
            <div className="zr-span-label">
              <span className="zr-leg-dot" style={{background:col}}/>
              <span className="zr-span-name">{s.organ}</span>
            </div>
            <div className="zr-span-track">
              <div className="zr-span-range" style={{left:`${pct(s.min)}%`, width:`${pct(s.max-s.min)}%`, background:`${col}33`}}/>
              <div className="zr-span-iqr"   style={{left:`${pct(s.q1)}%`,  width:`${pct(s.q3-s.q1)}%`,   background:col, opacity:.7}}/>
              <div className="zr-span-med"   style={{left:`${pct(s.median)}%`, background:col}}/>
            </div>
            <div className="zr-span-nums">
              <span>{zt.min} <b>{s.min}</b></span>
              <span>{zt.mean} <b>{s.mean}</b></span>
              <span>{zt.max} <b>{s.max}</b></span>
            </div>
          </div>
        );
      })}
      <div className="zr-span-axis">
        {[0,10,20,30,40,50].map(v => (
          <span key={v} style={{left:`${(v/maxSpan*100).toFixed(1)}%`}}>{v}</span>
        ))}
      </div>
      <p className="zr-note">{zt.spanNote}</p>
    </div>
  );
}

function ZRangeMatrix({ data, t }) {
  const zt = t.zrange;
  return (
    <div className="zr-matrix-wrap">
      <div className="zr-matrix-header">
        <span className="zr-matrix-pid-head"/>
        {ORGAN_ORDER.map(o => (
          <span key={o} className="zr-matrix-col-head" title={o}>
            {o.replace('_',' ').replace('OpticNerve','ON').replace('Submandibular','Sub')}
          </span>
        ))}
      </div>
      {data.map(patient => (
        <div key={patient.id} className="zr-matrix-row">
          <span className="zr-matrix-pid">{patient.id.replace('0522c','')}</span>
          {ORGAN_ORDER.map(organ => {
            const o = patient.organs[organ];
            const present = o?.present;
            return (
              <span key={organ}
                title={present ? `${organ}: z ${o.z_min}–${o.z_max}` : `${organ}: ${zt.matrixNote.includes('✕') ? 'absent' : 'yok'}`}
                className="zr-matrix-cell"
                style={{background: present ? ORGAN_HEX[organ] : '#1e293b', opacity: present ? 0.85 : 1}}>
                {!present && <span className="zr-matrix-x">✕</span>}
              </span>
            );
          })}
        </div>
      ))}
      <p className="zr-note">{zt.matrixNote}</p>
    </div>
  );
}

/* ── 2D Slice viewer ─────────────────────────────────────── */

function SliceViewerToggle({ src, patientId, t }) {
  const [open, setOpen] = useState(false);
  const s = t.slice;

  return (
    <div className="slice-viewer-wrap">
      <button
        className={`slice-viewer-btn${open ? ' slice-viewer-btn-open' : ''}`}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="slice-viewer-btn-icon">{open ? '▲' : '▼'}</span>
        {s.btn}
        <span className="slice-viewer-btn-sub">
          {src ? s.sub : s.generating}
        </span>
      </button>

      {open && (
        <div className="slice-viewer-panel">
          {src ? (
            <iframe
              key={patientId}
              title={`2D Slice — ${patientId}`}
              src={src}
              className="slice-viewer-iframe"
              loading="lazy"
            />
          ) : (
            <div className="slice-viewer-pending">
              <p>{s.notReady}</p>
              <code>.venv/bin/python "2- uNET/export_slice_viewer.py" --pddca-root .</code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Static pages ────────────────────────────────────────── */

function DatasetPage({ t }) {
  const d = t.dataset;
  return (
    <div className="static-page">
      <h2 className="static-page-title">{d.title}</h2>
      <p className="static-page-lead"><strong>Public Domain Database for Computational Anatomy (PDDCA)</strong>, {d.lead}</p>

      <div className="static-section">
        <h3 className="static-section-title">{d.generalInfo}</h3>
        <table className="static-table">
          <tbody>
            <tr><td>{d.modalite}</td><td>{d.modaliteVal}</td></tr>
            <tr><td>{d.totalPatient}</td><td>{d.totalPatientVal}</td></tr>
            <tr><td>{d.testOffsite}</td><td>{d.testOffsiteVal}</td></tr>
            <tr><td>{d.imageSize}</td><td>{d.imageSizeVal}</td></tr>
            <tr><td>{d.voxelSpacing}</td><td>{d.voxelSpacingVal}</td></tr>
            <tr><td>{d.license}</td><td>{d.licenseVal}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="static-section">
        <h3 className="static-section-title">{d.structures}</h3>
        <ul className="static-organ-list">
          {d.organs.map(([name, color]) => (
            <li key={name} className="static-organ-item">
              <span className="static-organ-dot" style={{ background: color }} />
              {name}
            </li>
          ))}
        </ul>
      </div>

      <div className="static-section">
        <h3 className="static-section-title">{d.dataSplit}</h3>
        <table className="static-table">
          <thead><tr><th>{d.splitSet}</th><th>{d.splitCount}</th><th>{d.splitUse}</th></tr></thead>
          <tbody>
            <tr><td>{d.train}</td><td>{d.trainCount}</td><td>{d.trainUse}</td></tr>
            <tr><td>{d.val}</td><td>{d.valCount}</td><td>{d.valUse}</td></tr>
            <tr><td>{d.testIn}</td><td>{d.testInCount}</td><td>{d.testInUse}</td></tr>
            <tr><td>{d.testOff}</td><td>{d.testOffCount}</td><td>{d.testOffUse}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="static-section">
        <h3 className="static-section-title">{d.source}</h3>
        <p>
          Raudaschl P.F. et al., "Evaluation of segmentation methods on head and neck CT: Auto‑segmentation
          challenge 2015," <em>Medical Physics</em>, 2017.{' '}
          <a href="https://www.imagenglab.com/newsite/pddca/" target="_blank" rel="noreferrer" className="static-link">
            imagenglab.com/newsite/pddca
          </a>
        </p>
      </div>
    </div>
  );
}

/* Specialized Chiasm+Optic model — static data */
const NSLICE_PATIENTS = [
  { id:'0522c0555', Chiasm:12.4, OpticNerve_L:42.2, OpticNerve_R:57.3 },
  { id:'0522c0576', Chiasm: 0.9, OpticNerve_L:57.1, OpticNerve_R:69.3 },
  { id:'0522c0598', Chiasm:43.5, OpticNerve_L:73.9, OpticNerve_R:67.2 },
  { id:'0522c0659', Chiasm:58.3, OpticNerve_L:61.5, OpticNerve_R:59.5 },
  { id:'0522c0661', Chiasm:14.4, OpticNerve_L:59.6, OpticNerve_R:74.9 },
  { id:'0522c0667', Chiasm:18.0, OpticNerve_L:55.0, OpticNerve_R:49.2 },
  { id:'0522c0669', Chiasm:40.8, OpticNerve_L:54.5, OpticNerve_R:48.2 },
  { id:'0522c0708', Chiasm:51.8, OpticNerve_L:37.0, OpticNerve_R:62.0 },
  { id:'0522c0727', Chiasm:64.7, OpticNerve_L:72.1, OpticNerve_R:63.3 },
  { id:'0522c0746', Chiasm:38.5, OpticNerve_L:57.6, OpticNerve_R:50.0 },
];
const NSLICE_ORGANS = ['Chiasm','OpticNerve_L','OpticNerve_R'];
const NSLICE_MEANS  = { Chiasm:34.33, OpticNerve_L:57.05, OpticNerve_R:60.10 };
// Best U-Net 2.5D v2 scores for same organs (for comparison)
const UNET25D_BEST  = { Chiasm:51.20, OpticNerve_L:58.85, OpticNerve_R:61.98 };

function NSliceResultsTable({ t }) {
  const [showPatients, setShowPatients] = useState(false);
  const lang = t === TRANSLATIONS.en ? 'en' : 'tr';

  return (
    <div className="study-results-wrap">
      <p className="study-results-note">
        {lang === 'tr'
          ? '10 test hastası ortalaması. U-Net 2.5D v2 (genel model) ile karşılaştırma.'
          : 'Mean over 10 test patients. Comparison with U-Net 2.5D v2 (general model).'}
      </p>

      {/* Comparison mini table */}
      <div className="study-results-table-wrap">
        <table className="study-results-table">
          <thead>
            <tr>
              <th>{lang === 'tr' ? 'Organ' : 'Organ'}</th>
              <th className="study-results-th-best">{lang === 'tr' ? 'Uzmanlaşmış Model' : 'Specialized Model'}</th>
              <th>U-Net 2.5D v2</th>
            </tr>
          </thead>
          <tbody>
            {NSLICE_ORGANS.map((organ) => {
              const specialized = NSLICE_MEANS[organ];
              const general     = UNET25D_BEST[organ];
              const specBetter  = specialized >= general;
              return (
                <tr key={organ}>
                  <td className="study-results-organ">{organ}</td>
                  <td className={`study-results-val${specBetter ? ' study-results-val-best' : ''}`}
                      style={diceCellStyle(specialized)}>
                    {specialized.toFixed(2)}
                  </td>
                  <td className={`study-results-val${!specBetter ? ' study-results-val-best' : ''}`}
                      style={diceCellStyle(general)}>
                    {general.toFixed(2)}
                  </td>
                </tr>
              );
            })}
            <tr className="study-results-mean-row">
              <td className="study-results-organ"><strong>{lang === 'tr' ? 'Ortalama' : 'Mean'}</strong></td>
              <td className="study-results-val" style={diceCellStyle(50.49)}><strong>50.49</strong></td>
              <td className="study-results-val" style={diceCellStyle(57.34)}><strong>57.34</strong></td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Per-patient toggle */}
      <button className="nslice-expand-btn" onClick={() => setShowPatients(v => !v)}>
        {showPatients
          ? (lang === 'tr' ? '▲ Hasta detaylarını gizle' : '▲ Hide per-patient details')
          : (lang === 'tr' ? '▼ Hasta bazında sonuçları göster' : '▼ Show per-patient results')}
      </button>

      {showPatients && (
        <div className="study-results-table-wrap" style={{marginTop:'0.6rem'}}>
          <table className="study-results-table">
            <thead>
              <tr>
                <th>{lang === 'tr' ? 'Hasta' : 'Patient'}</th>
                {NSLICE_ORGANS.map(o => <th key={o}>{o}</th>)}
                <th>{lang === 'tr' ? 'Ort.' : 'Avg'}</th>
              </tr>
            </thead>
            <tbody>
              {NSLICE_PATIENTS.map(p => {
                const avg = NSLICE_ORGANS.reduce((s, o) => s + p[o], 0) / NSLICE_ORGANS.length;
                return (
                  <tr key={p.id}>
                    <td className="study-results-organ" style={{fontFamily:'monospace',fontSize:'0.78rem'}}>
                      {p.id.replace('0522c','')}
                    </td>
                    {NSLICE_ORGANS.map(o => (
                      <td key={o} className="study-results-val" style={diceCellStyle(p[o])}>
                        {p[o].toFixed(1)}
                      </td>
                    ))}
                    <td className="study-results-val" style={diceCellStyle(avg)}>
                      {avg.toFixed(1)}
                    </td>
                  </tr>
                );
              })}
              <tr className="study-results-mean-row">
                <td className="study-results-organ"><strong>{lang === 'tr' ? 'Ort.' : 'Mean'}</strong></td>
                {NSLICE_ORGANS.map(o => (
                  <td key={o} className="study-results-val" style={diceCellStyle(NSLICE_MEANS[o])}>
                    <strong>{NSLICE_MEANS[o].toFixed(2)}</strong>
                  </td>
                ))}
                <td className="study-results-val" style={diceCellStyle(50.49)}><strong>50.49</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* U-Net results — static data from notebooks */
const UNET_ORGANS = [
  'BrainStem','Chiasm','Mandible',
  'OpticNerve_L','OpticNerve_R',
  'Parotid_L','Parotid_R',
  'Submandibular_L','Submandibular_R',
];

const UNET_METHODS = [
  { key: 'unet3d',  label: 'U-Net 3D',               desc: '3D patch 64³ · 1.5 mm izotropik · 100 epoch' },
  { key: 'unet25d', label: 'U-Net 2.5D v2',           desc: '3-slice giriş · 128×128 ROI · TTA · 300 epoch' },
  { key: 'r1',      label: 'Reçete 1 (nnU-Net tarzı)', desc: '~1.5 mm izotropik · elastik aug · 150 epoch' },
  { key: 'r2',      label: 'Reçete 2',                desc: 'notebook-5 recipe2 preprocess · 150 epoch' },
  { key: 'r3',      label: 'Reçete 3 TAM',            desc: 'Kaba kırpma + 2.5D + 64×64 ROI + oversampling · 150 epoch' },
];

const UNET_DATA = {
  BrainStem:       { unet3d:83.33, unet25d:84.35, r1:83.62, r2:83.10, r3:69.24 },
  Chiasm:          { unet3d:44.22, unet25d:51.20, r1:32.90, r2:45.77, r3:39.52 },
  Mandible:        { unet3d:84.52, unet25d:92.34, r1:89.66, r2:86.30, r3:88.80 },
  OpticNerve_L:    { unet3d:28.36, unet25d:58.85, r1:42.58, r2:56.64, r3:58.03 },
  OpticNerve_R:    { unet3d:25.74, unet25d:61.98, r1:47.04, r2:37.23, r3:55.01 },
  Parotid_L:       { unet3d:47.97, unet25d:82.99, r1:77.23, r2:81.08, r3:78.37 },
  Parotid_R:       { unet3d:48.85, unet25d:81.31, r1:82.49, r2:77.08, r3:75.71 },
  Submandibular_L: { unet3d:27.15, unet25d:63.81, r1:69.96, r2:73.65, r3:70.48 },
  Submandibular_R: { unet3d:29.19, unet25d:68.11, r1:73.86, r2:73.12, r3:77.03 },
};

function UNetResultsTable({ t }) {
  const s = t.studies;
  const colMeans = UNET_METHODS.map(({ key }) => {
    const vals = UNET_ORGANS.map(o => UNET_DATA[o][key]).filter(v => v != null);
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  });
  const bestColIdx = colMeans.reduce((bi, v, i) => (v > colMeans[bi] ? i : bi), 0);

  return (
    <div className="study-results-wrap">
      <p className="study-results-note">{s.unetResultsNote}</p>

      {/* Method description mini-cards */}
      <div className="unet-method-cards">
        {UNET_METHODS.map((m) => (
          <div key={m.key} className="unet-method-card">
            <span className="unet-method-card-title">{m.label}</span>
            <span className="unet-method-card-desc">{m.desc}</span>
          </div>
        ))}
      </div>

      <div className="study-results-table-wrap">
        <table className="study-results-table">
          <thead>
            <tr>
              <th>{s.organ ?? 'Organ'}</th>
              {UNET_METHODS.map((m, i) => (
                <th key={m.key} className={i === bestColIdx ? 'study-results-th-best' : ''} title={m.desc}>
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {UNET_ORGANS.map((organ) => {
              const row = UNET_DATA[organ];
              const bestVal = Math.max(...UNET_METHODS.map(m => row[m.key] ?? -1));
              return (
                <tr key={organ}>
                  <td className="study-results-organ">{organ}</td>
                  {UNET_METHODS.map((m) => {
                    const v = row[m.key];
                    const isBest = v === bestVal;
                    return (
                      <td key={m.key}
                        className={`study-results-val${isBest ? ' study-results-val-best' : ''}`}
                        style={diceCellStyle(v)}>
                        {v != null ? v.toFixed(2) : '—'}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
            <tr className="study-results-mean-row">
              <td className="study-results-organ"><strong>Ortalama</strong></td>
              {colMeans.map((v, i) => (
                <td key={i}
                  className={`study-results-val${i === bestColIdx ? ' study-results-val-best' : ''}`}
                  style={diceCellStyle(v)}>
                  <strong>{v.toFixed(2)}</strong>
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AtlasResultsTable({ atlasDiceDoc, t }) {
  const s = t.studies;
  if (!atlasDiceDoc?.methods?.length) return null;

  const organs = atlasDiceDoc.structureNames || [];
  const methods = atlasDiceDoc.methods;

  // compute per-organ mean Dice across available patients for each method
  const means = methods.map((m) => {
    const patients = Object.values(m.diceByPatient || {});
    if (!patients.length) return {};
    return Object.fromEntries(
      organs.map((organ) => {
        const vals = patients.map((p) => p[organ]).filter((v) => v != null);
        const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
        return [organ, avg != null ? avg * 100 : null];
      })
    );
  });

  // find best method index per organ
  const bestIdx = organs.map((organ) => {
    let best = -1, bestVal = -1;
    means.forEach((m, i) => {
      if ((m[organ] ?? -1) > bestVal) { bestVal = m[organ]; best = i; }
    });
    return best;
  });

  // compute column means
  const colMeans = means.map((m) => {
    const vals = organs.map((o) => m[o]).filter((v) => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  });
  const bestColIdx = colMeans.reduce((bi, v, i) => ((v ?? -1) > (colMeans[bi] ?? -1) ? i : bi), 0);

  const nPatients = Object.keys(methods[0]?.diceByPatient || {}).length;
  const methodLabels = ['HQ Deformable', 'HQ Deformable + MV', 'HQ Deformable + STAPLE'];

  return (
    <div className="study-results-wrap">
      <h4 className="study-results-title">{s.resultsTitle}</h4>
      <p className="study-results-note">{s.resultsNote} (n = {nPatients})</p>
      <div className="study-results-table-wrap">
        <table className="study-results-table">
          <thead>
            <tr>
              <th>{s.organ}</th>
              {methodLabels.map((l, i) => (
                <th key={i} className={i === bestColIdx ? 'study-results-th-best' : ''}>{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {organs.map((organ, oi) => (
              <tr key={organ}>
                <td className="study-results-organ">{organ}</td>
                {means.map((m, mi) => {
                  const v = m[organ];
                  const isBest = bestIdx[oi] === mi;
                  return (
                    <td key={mi}
                      className={`study-results-val${isBest ? ' study-results-val-best' : ''}`}
                      style={diceCellStyle(v)}>
                      {v != null ? v.toFixed(1) : '—'}
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr className="study-results-mean-row">
              <td className="study-results-organ"><strong>Ortalama</strong></td>
              {colMeans.map((v, i) => (
                <td key={i}
                  className={`study-results-val${i === bestColIdx ? ' study-results-val-best' : ''}`}
                  style={diceCellStyle(v)}>
                  <strong>{v != null ? v.toFixed(1) : '—'}</strong>
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StudiesPage({ t, atlasDiceDoc }) {
  const s = t.studies;
  return (
    <div className="static-page">
      <h2 className="static-page-title">{s.title}</h2>
      <p className="static-page-lead">{s.lead}</p>

      {/* ── Atlas (önce) ── */}
      <div className="static-section">
        <div className="study-card">
          <div className="study-card-header study-card-header-atlas">
            <span className="study-card-icon">🗺️</span>
            <h3 className="study-card-title">{s.atlasTitle}</h3>
          </div>
          <div className="study-card-body">
            <p>{s.atlasDesc}</p>
            <table className="static-table">
              <tbody>
                <tr><td>{s.regLib}</td><td>{s.regLibVal}</td></tr>
                <tr><td>{s.regType}</td><td>{s.regTypeVal}</td></tr>
                <tr><td>{s.numAtlas}</td><td>{s.numAtlasVal}</td></tr>
              </tbody>
            </table>
            <h4 className="study-method-subtitle">{s.fusionMethods}</h4>
            <ul className="study-method-list">
              <li><strong>{s.hqDef}</strong> — {s.hqDefDesc}</li>
              <li><strong>{s.hqMV}</strong> — {s.hqMVDesc}</li>
              <li><strong>{s.hqSTAPLE}</strong> — {s.hqSTAPLEDesc}</li>
            </ul>

            <AtlasResultsTable atlasDiceDoc={atlasDiceDoc} t={t} />
          </div>
        </div>
      </div>

      {/* ── U-Net (sonra) ── */}
      <div className="static-section">
        <div className="study-card">
          <div className="study-card-header study-card-header-unet">
            <span className="study-card-icon">🧠</span>
            <h3 className="study-card-title">{s.unetTitle}</h3>
          </div>
          <div className="study-card-body">
            <p>{s.unetDesc}</p>
            <UNetResultsTable t={t} />
          </div>
        </div>
      </div>

      {/* ── Uzmanlaşmış model (Chiasm + Optik sinir) ── */}
      <div className="static-section">
        <div className="study-card">
          <div className="study-card-header study-card-header-special">
            <span className="study-card-icon">👁️</span>
            <h3 className="study-card-title">{s.specialTitle}</h3>
          </div>
          <div className="study-card-body">
            <p>{s.specialDesc}</p>
            <table className="static-table" style={{marginBottom:'0.75rem'}}>
              <tbody>
                {s.specialDetails.map(([k, v]) => (
                  <tr key={k}><td>{k}</td><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
            <NSliceResultsTable t={t} />
          </div>
        </div>
      </div>

      <div className="static-section">
        <h3 className="static-section-title">{s.metricTitle}</h3>
        <p>{s.metricDesc}</p>
      </div>
    </div>
  );
}

function ContactPage({ t }) {
  const c = t.contact;
  return (
    <div className="static-page">
      <h2 className="static-page-title">{c.title}</h2>
      <p className="static-page-lead">{c.lead}</p>

      <div className="static-section contact-cards-grid">
        {/* Olcay Çoban */}
        <div className="contact-card">
          <div className="contact-avatar">OÇ</div>
          <div className="contact-info">
            <h3 className="contact-name">Olcay Çoban</h3>
            <ul className="contact-list">
              <li className="contact-item">
                <span className="contact-icon contact-icon-email">✉</span>
                <a href="mailto:olcay.coban@iletisim.gov.tr" className="static-link">
                  olcay.coban@iletisim.gov.tr
                </a>
              </li>
              <li className="contact-item">
                <span className="contact-icon contact-icon-linkedin">in</span>
                <a href="https://www.linkedin.com/in/olcay-%C3%A7oban-57084416b/" target="_blank" rel="noreferrer" className="static-link">
                  linkedin.com/in/olcay-çoban
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Cemil Zalluhoğlu */}
        <div className="contact-card">
          <div className="contact-avatar contact-avatar-alt">CZ</div>
          <div className="contact-info">
            <h3 className="contact-name">Doç. Dr. Cemil Zalluhoğlu</h3>
            <ul className="contact-list">
              <li className="contact-item">
                <span className="contact-icon contact-icon-email">✉</span>
                <a href="mailto:cemil@cs.hacettepe.edu.tr" className="static-link">
                  cemil@cs.hacettepe.edu.tr
                </a>
              </li>
              <li className="contact-item">
                <span className="contact-icon contact-icon-linkedin">in</span>
                <a href="https://www.linkedin.com/in/cemilzalluhoglu/" target="_blank" rel="noreferrer" className="static-link">
                  linkedin.com/in/cemilzalluhoglu
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Main App ────────────────────────────────────────────── */

export default function App() {
  const [page, setPage] = useState('viewer');
  const [lang, setLang] = useState('tr');
  const [darkMode, setDarkMode] = useState(false);
  const [manifest, setManifest] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [selectedOrgan, setSelectedOrgan] = useState(null);
  const [atlasManifest, setAtlasManifest] = useState(null);
  const [atlasDiceDoc, setAtlasDiceDoc] = useState(null);
  const [atlasPairwiseDoc, setAtlasPairwiseDoc] = useState(null);
  const [atlasMetaDoc, setAtlasMetaDoc] = useState(null);
  const [selectedItems, setSelectedItems] = useState(['gt', 'unet', 'deformable', 'majority_voting', 'staple']);

  const t = TRANSLATIONS[lang];

  function toggleViewerItem(id) {
    setSelectedItems((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
  }

  function getSceneUrl(id, pid, mf, af) {
    const p = mf?.patients?.find((pt) => pt.id === pid);
    if (id === 'gt')      return p?.gt3d   ? `${assetUrl(p.gt3d)}?cb=${SCENE3D_CACHE_BUST}`   : null;
    if (id === 'unet')    return p?.pred3d  ? `${assetUrl(p.pred3d)}?cb=${SCENE3D_CACHE_BUST}`  : null;
    if (id === 'unet_fn') return p?.fn3d    ? `${assetUrl(p.fn3d)}?cb=${SCENE3D_CACHE_BUST}`    : null;
    if (id === 'unet_fp') return p?.fp3d    ? `${assetUrl(p.fp3d)}?cb=${SCENE3D_CACHE_BUST}`    : null;
    const m = af?.methods?.find((m) => m.id === id);
    const rel = m?.scenes?.[pid];
    return rel ? `${assetUrl(rel)}?cb=${SCENE_ATLAS_CACHE_BUST}` : null;
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${assetUrl('pddca-viz/manifest.json')}?cb=${MANIFEST_CACHE_BUST}`);
        if (!res.ok) throw new Error(`manifest ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        setManifest(data);
        if (data.patients?.length) setPatientId((prev) => prev || data.patients[0].id);
      } catch (e) {
        if (!cancelled) setLoadError(String(e.message || e));
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${assetUrl('pddca-viz-atlas/manifest.json')}?cb=${ATLAS_MANIFEST_CACHE_BUST}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setAtlasManifest(data);
      } catch { if (!cancelled) setAtlasManifest(null); }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${assetUrl('pddca-viz-atlas/atlas_dice_by_method.json')}?cb=${ATLAS_DICE_CACHE_BUST}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setAtlasDiceDoc(data);
      } catch { if (!cancelled) setAtlasDiceDoc(null); }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${assetUrl('pddca-viz-atlas/atlas_deformable_pairwise.json')}?cb=${ATLAS_PAIRWISE_CACHE_BUST}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setAtlasPairwiseDoc(data);
      } catch { if (!cancelled) setAtlasPairwiseDoc(null); }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${assetUrl('pddca-viz-atlas/atlas_meta.json')}?cb=${ATLAS_META_CACHE_BUST}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setAtlasMetaDoc(data);
      } catch { if (!cancelled) setAtlasMetaDoc(null); }
    })();
    return () => { cancelled = true; };
  }, []);

  const current = useMemo(
    () => manifest?.patients?.find((p) => p.id === patientId),
    [manifest, patientId]
  );

  useEffect(() => { setSelectedOrgan(null); }, [patientId]);

  const NAV_ITEMS = [
    { id: 'viewer',  label: t.nav.viewer  },
    { id: 'zranges', label: t.nav.zranges },
    { id: 'dataset', label: t.nav.dataset },
    { id: 'studies', label: t.nav.studies },
    { id: 'contact', label: t.nav.contact },
  ];

  return (
    <div className={`app${darkMode ? ' dark' : ''}`}>
      <header className="header">
        <div className="header-top">
          <h1>{t.siteTitle}</h1>
          <p className="header-sub">{t.siteSub}</p>
        </div>
        <div className="header-controls-row">
          <nav className="nav">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                className={`nav-btn${page === item.id ? ' nav-btn-active' : ''}`}
                onClick={() => setPage(item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <div className="header-controls">
            <button
              className="ctrl-btn"
              onClick={() => setLang((l) => l === 'tr' ? 'en' : 'tr')}
              title="Toggle language / Dil değiştir"
            >
              {lang === 'tr' ? '🇬🇧 EN' : '🇹🇷 TR'}
            </button>
            <button
              className="ctrl-btn"
              onClick={() => setDarkMode((d) => !d)}
              title={darkMode ? t.lightMode : t.darkMode}
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
      </header>

      {page === 'zranges' && <ZRangePage t={t}/>}
      {page === 'dataset' && <DatasetPage t={t}/>}
      {page === 'studies' && <StudiesPage t={t} atlasDiceDoc={atlasDiceDoc}/>}
      {page === 'contact' && <ContactPage t={t}/>}

      {page === 'viewer' && loadError && (
        <div className="banner banner-error">
          {t.manifestError} {loadError}
        </div>
      )}

      {page === 'viewer' && manifest && (
        <div className="app-body">
          {/* Sol sidebar */}
          <aside className="viewer-sidebar">
            <div className="viewer-sidebar-section">
              <label className="viewer-sidebar-label" htmlFor="patient-select">{t.sidebar.patient}</label>
              <select
                id="patient-select"
                className="viewer-sidebar-select"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
              >
                {manifest.patients.map((p) => (
                  <option key={p.id} value={p.id}>{p.id}</option>
                ))}
              </select>
            </div>

            <div className="viewer-sidebar-divider" />

            <div className="viewer-sidebar-section">
              <p className="viewer-sidebar-label">{t.sidebar.views}</p>
              <ul className="viewer-sidebar-list">
                {VIEWER_ITEMS.map((item) => {
                  const scene = getSceneUrl(item.id, patientId, manifest, atlasManifest);
                  const isReady = !!scene;
                  const isChecked = selectedItems.includes(item.id);
                  return (
                    <li key={item.id} className="viewer-sidebar-item">
                      <label className={`viewer-sidebar-item-label${isChecked ? ' viewer-sidebar-item-label-on' : ''}`}>
                        <input
                          type="checkbox"
                          className="viewer-sidebar-checkbox"
                          checked={isChecked}
                          onChange={() => toggleViewerItem(item.id)}
                        />
                        <span className="viewer-sidebar-item-name">{item.label}</span>
                        <span className={`viewer-sidebar-badge ${isReady ? 'viewer-sidebar-badge-ready' : 'viewer-sidebar-badge-pending'}`}>
                          {isReady ? t.sidebar.ready : t.sidebar.pending}
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
            </div>

            <div className="viewer-sidebar-divider" />
            <p className="viewer-sidebar-note">{t.sidebar.note}</p>
          </aside>

          {/* Sağ ana alan */}
          <div className="app-main">
            {current && (
              <section className="slice-viewer-section">
                <SliceViewerToggle
                  src={current.sliceViewer ? `${assetUrl(current.sliceViewer)}?cb=${MANIFEST_CACHE_BUST}` : null}
                  patientId={current.id}
                  t={t}
                />
              </section>
            )}

            {current && selectedItems.length > 0 && (
              <section className="viewer-grid-section">
                <div className="viewer-grid" style={{ '--vcol': Math.min(selectedItems.length, 2) }}>
                  {VIEWER_ITEMS.filter((item) => selectedItems.includes(item.id)).map((item) => {
                    const scene = getSceneUrl(item.id, patientId, manifest, atlasManifest);
                    return (
                      <div key={item.id} className="viewer-grid-col">
                        <h3 className="viewer-grid-col-title">{item.shortLabel}</h3>
                        {scene ? (
                          <figure className="viewer-grid-figure">
                            <div className="viewer-grid-iframe-wrap">
                              <iframe
                                key={`${patientId}-${item.id}`}
                                title={`${item.label} — ${patientId}`}
                                src={scene}
                                className="viewer-grid-iframe"
                                style={{ height: item.iframeHeight + 'px' }}
                                loading="lazy"
                              />
                            </div>
                          </figure>
                        ) : (
                          <div className="atlas-3d-generate-hint">
                            <p className="atlas-3d-generate-title">{t.sceneNotReady}</p>
                            <p className="atlas-3d-generate-note">{t.sceneNotReadyNote}</p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
            )}

            {current && (
              <>
                <DicePanel
                  patientId={current.id}
                  structureNames={manifest.structureNames}
                  organMetrics={current.organMetrics}
                  dicePercent={current.dicePercent}
                  selectedOrgan={selectedOrgan}
                  onSelectOrgan={setSelectedOrgan}
                  t={t}
                />

                {atlasDiceDoc ? (
                  <AtlasDiceTriplet
                    patientId={current.id}
                    diceDoc={atlasDiceDoc}
                    structureNames={manifest.structureNames}
                    pairwiseDoc={atlasPairwiseDoc}
                    atlasMeta={atlasMetaDoc}
                    t={t}
                  />
                ) : null}
              </>
            )}
          </div>
        </div>
      )}

      {page === 'viewer' && <AtlasArtifactsFooter t={t}/>}
    </div>
  );
}
