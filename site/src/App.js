import { useEffect, useMemo, useState } from 'react';
import './App.css';

const publicBase = process.env.PUBLIC_URL || '';
/** Tarayıcı eski scene_3d.html önbelleğinden vermesin (export sonrası yenile). */
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
  return {
    background: `hsl(${hue}, 72%, 88%)`,
    color: '#0f172a',
    fontWeight: 600,
  };
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

function DiceDetailPanel({ organName, organMetrics, dicePercent }) {
  if (!organName) {
    return (
      <div className="dice-detail dice-detail-placeholder">
        <p className="dice-detail-hint">Soldan bir organ adına tıklayın; Dice ve vokseller sağda görünür.</p>
      </div>
    );
  }

  const m = organMetrics?.[organName];
  const fallbackPct = dicePercent?.[organName];

  if (!m && fallbackPct == null) {
    return (
      <div className="dice-detail">
        <h3 className="dice-detail-title">{organName}</h3>
        <p>Veri yok.</p>
      </div>
    );
  }

  if (!m) {
    return (
      <div className="dice-detail">
        <h3 className="dice-detail-title">{organName}</h3>
        <p className="dice-detail-dsc">
          Dice (DSC): <strong>{fallbackPct.toFixed(1)} %</strong>
        </p>
        <p className="dice-detail-upgrade">
          Bu tablo için manifest’e <code>organMetrics</code> gerekir: mevcut{' '}
          <code>scene_3d.html</code> dosyalarından{' '}
          <code>fill_manifest_organ_metrics_from_scene3d.py</code> çalıştırın (model gerekmez), veya{' '}
          <code>fill_manifest_dice.py</code> / tam export.
        </p>
      </div>
    );
  }

  const {
    dicePercent: dsc,
    gtVoxels,
    predVoxels,
    tpVoxels,
    fnVoxels,
    fpVoxels,
  } = m;
  const empty = (gtVoxels ?? 0) === 0 && (predVoxels ?? 0) === 0;

  if (empty) {
    return (
      <div className="dice-detail">
        <h3 className="dice-detail-title">{organName}</h3>
        <p>Bu vakada hem GT hem tahminde bu organ yok (0 vokseller).</p>
      </div>
    );
  }

  const gt = gtVoxels ?? 0;
  const pred = predVoxels ?? 0;
  const tp = tpVoxels ?? 0;
  const fn = fnVoxels ?? 0;
  const fp = fpVoxels ?? 0;
  const denom = gt + pred;
  const sumParts = 2 * tp + fn + fp;
  const formulaGtPred =
    denom > 0
      ? `200 × ${fmtInt(tp)} / (${fmtInt(gt)} + ${fmtInt(pred)})`
      : '';
  const formulaTpFnFp =
    sumParts > 0
      ? `100 × (2 × ${fmtInt(tp)}) / (2 × ${fmtInt(tp)} + ${fmtInt(fn)} + ${fmtInt(fp)})`
      : '';

  return (
    <div className="dice-detail">
      <h3 className="dice-detail-title">{organName}</h3>
      <p className="dice-detail-dsc">
        Dice (DSC):{' '}
        <strong>{dsc != null ? `${dsc.toFixed(1)} %` : '—'}</strong>
      </p>
      <p className="dice-detail-intro">
        Aşağıdaki sayılar, üstteki 3B sahnesinde bu organ için gördüğünüz dört panelle aynıdır (fareyle
        vokseller).
      </p>
      <ul className="dice-detail-quadrants">
        <li>
          <strong>Ground truth (3D)</strong> → GT toplamı: <strong>{fmtInt(gt)}</strong>
        </li>
        <li>
          <strong>Model tahmini (3D)</strong> → Tahmin toplamı: <strong>{fmtInt(pred)}</strong>
        </li>
        <li>
          <strong>GT’de olup tahminle uyuşmayan</strong> (GT etiketi) → FN:{' '}
          <strong>{fmtInt(fn)}</strong>
        </li>
        <li>
          <strong>Tahminde olup GT ile uyuşmayan</strong> (tahmin etiketi) → FP:{' '}
          <strong>{fmtInt(fp)}</strong>
        </li>
      </ul>
      <p className="dice-detail-tpderiv">
        <strong>Kesişim (TP)</strong>, hem GT’den FN çıkarılarak hem tahminden FP çıkarılarak bulunur:{' '}
        <span className="dice-detail-mono">
          TP = GT − FN = {fmtInt(gt)} − {fmtInt(fn)} = {fmtInt(tp)}
        </span>
        {' · '}
        <span className="dice-detail-mono">
          TP = Tahmin − FP = {fmtInt(pred)} − {fmtInt(fp)} = {fmtInt(tp)}
        </span>
      </p>
      <dl className="dice-detail-dl dice-detail-dl-compact">
        <dt>GT vokseller</dt>
        <dd>{fmtInt(gt)}</dd>
        <dt>Tahmin vokseller</dt>
        <dd>{fmtInt(pred)}</dd>
        <dt>Kesişim (TP)</dt>
        <dd>{fmtInt(tp)}</dd>
        <dt>FN</dt>
        <dd>{fmtInt(fn)}</dd>
        <dt>FP</dt>
        <dd>{fmtInt(fp)}</dd>
      </dl>
      <p className="dice-detail-identity">
        Payda için: <span className="dice-detail-mono">GT + Tahmin = 2×TP + FN + FP</span> →{' '}
        <strong>{fmtInt(denom)}</strong> = <strong>{fmtInt(sumParts)}</strong>
        {denom !== sumParts && (
          <span className="dice-detail-warn"> (küçük fark sayım yuvarlaması olabilir)</span>
        )}
        .
      </p>
      {formulaGtPred && dsc != null && (
        <div className="dice-detail-formulas">
          <p className="dice-detail-formula">
            <strong>DSC% (GT + tahmin paydası):</strong> {formulaGtPred} ≈{' '}
            <strong>{dsc.toFixed(1)}</strong>
          </p>
          {formulaTpFnFp && (
            <p className="dice-detail-formula">
              <strong>Aynı sonuç (TP + FN + FP paydası):</strong> {formulaTpFnFp} ≈{' '}
              <strong>{dsc.toFixed(1)}</strong>
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function DicePanel({
  patientId,
  structureNames,
  organMetrics,
  dicePercent,
  selectedOrgan,
  onSelectOrgan,
}) {
  const hasAny =
    (organMetrics && Object.keys(organMetrics).length > 0) ||
    (dicePercent && Object.keys(dicePercent).length > 0);
  if (!hasAny || !structureNames?.length) {
    return null;
  }

  return (
    <section className="dice-section dice-section-wide" aria-label="Dice skorları">
      <h2 className="dice-heading">
        Hasta <span className="dice-pid">{patientId}</span> · Dice (DSC, %)
      </h2>
      <p className="dice-note">
        Organ başına: 200 × kesişim / (GT vokseller + tahmin vokseller). Satıra tıklayınca sağda detay.
      </p>
      <div className="dice-split">
        <div className="dice-table-wrap">
          <table className="dice-table">
            <thead>
              <tr>
                <th scope="col">Organ</th>
                <th scope="col">DSC (%)</th>
              </tr>
            </thead>
            <tbody>
              {structureNames.map((name) => {
                const v = rowDice(organMetrics, dicePercent, name);
                const sel = selectedOrgan === name;
                return (
                  <tr
                    key={name}
                    className={sel ? 'dice-row dice-row-selected' : 'dice-row'}
                  >
                    <td className="dice-organ">
                      <button
                        type="button"
                        className="dice-organ-btn"
                        onClick={() => onSelectOrgan(name)}
                        aria-pressed={sel}
                      >
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
        <DiceDetailPanel
          organName={selectedOrgan}
          organMetrics={organMetrics}
          dicePercent={dicePercent}
        />
      </div>
    </section>
  );
}

/** Yerel repoda tutulan atlas deneyleri; `.gitignore` ile commit dışı olabilir. */
const ATLAS_CACHE_FILES = [
  {
    file: 'train25_test10_hq_deformable_cache.pkl',
    note: 'HQ deformable kayıt — önbellek',
  },
  {
    file: 'train25_test10_hq_majority_voting_deformable_cache.pkl',
    note: 'HQ deformable + çoğunluk oylaması — önbellek',
  },
  {
    file: 'train25_test10_hq_staple_deformable_cache.pkl',
    note: 'HQ deformable + STAPLE — önbellek',
  },
];

function AtlasDiceTriplet({ patientId, diceDoc, structureNames, pairwiseDoc, atlasMeta }) {
  const methods = diceDoc?.methods;
  const [deformableAtlas, setDeformableAtlas] = useState('mean');

  useEffect(() => {
    setDeformableAtlas('mean');
  }, [patientId]);

  if (!methods?.length) return null;

  const organs = structureNames || diceDoc.structureNames || [];
  const top5 = atlasMeta?.top5AtlasIdsForFusion || pairwiseDoc?.top5AtlasIdsByMeanDice || [];
  const atlasOrder = pairwiseDoc?.atlasIdsDisplayOrder || [];

  return (
    <section className="atlas-dice-triplet" aria-label="Atlas Dice üç yöntem">
      <h2 className="atlas-dice-triplet-heading">Atlas — üç yöntem (09 notebook ile uyumlu)</h2>
      <p className="atlas-dice-triplet-note">{diceDoc.note}</p>
      {top5.length > 0 && (
        <p className="atlas-dice-triplet-top5">
          <strong>En iyi 5 atlas</strong> (MV / STAPLE seçimi; <code>df_def</code> genel ortalama Dice):{' '}
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
                    Atlas seçimi (Affine+Deformable çiftleri)
                  </label>
                  <select
                    id="atlas-deformable-select"
                    className="atlas-dice-atlas-select"
                    value={deformableAtlas}
                    onChange={(e) => setDeformableAtlas(e.target.value)}
                  >
                    <option value="mean">25 atlas — organ başına ortalama Dice</option>
                    {atlasOrder.map((aid) => {
                      const isTop = top5.includes(aid);
                      return (
                        <option key={aid} value={aid}>
                          {aid}
                          {isTop ? ' (en iyi 5)' : ''}
                        </option>
                      );
                    })}
                  </select>
                </div>
              ) : null}
              {m.id === 'majority_voting' || m.id === 'staple' ? (
                <p className="atlas-dice-fusion-hint">
                  Notebook: <strong>en iyi 5 atlas</strong> ile birleşik tahmin; tek satır / organ.
                </p>
              ) : null}
              <table className="atlas-dice-mini-table">
                <thead>
                  <tr>
                    <th scope="col">Organ</th>
                    <th scope="col">DSC (%)</th>
                  </tr>
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



const VIEWER_ITEMS = [
  {
    id: 'gt',
    label: 'Ground Truth',
    shortLabel: 'GT',
    iframeHeight: 560,
  },
  {
    id: 'unet',
    label: 'U-Net tahmini',
    shortLabel: 'U-Net',
    iframeHeight: 560,
  },
  {
    id: 'unet_fn',
    label: 'U-Net FN',
    shortLabel: 'U-Net FN',
    iframeHeight: 560,
  },
  {
    id: 'unet_fp',
    label: 'U-Net FP',
    shortLabel: 'U-Net FP',
    iframeHeight: 560,
  },
  {
    id: 'deformable',
    label: 'HQ Deformable',
    shortLabel: 'Deformable',
    iframeHeight: 560,
  },
  {
    id: 'majority_voting',
    label: 'HQ Deformable + MV',
    shortLabel: 'MV',
    iframeHeight: 560,
  },
  {
    id: 'staple',
    label: 'HQ Deformable + STAPLE',
    shortLabel: 'STAPLE',
    iframeHeight: 560,
  },
];




function AtlasArtifactsFooter() {
  return (
    <footer className="project-refs" aria-label="Atlas tabanlı önbellek dosyaları">
      <h2 className="project-refs-title">Atlas önbellek dosya adları</h2>
      <p className="project-refs-intro">
        Dice özeti: <code className="project-refs-mono">export_atlas_pkl_3d.py</code>. Atlas 3B sahnesi ancak
        hacim pickle’ı / <code className="project-refs-mono">scene_3d.html</code> üretildiğinde görünür.
      </p>
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


/* ── Z-Range visualisation page ──────────────────────────── */

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

function ZRangePage() {
  const [data, setData] = useState(null);
  const [view, setView] = useState('gantt'); // gantt | span | matrix
  const [hoveredOrgan, setHoveredOrgan] = useState(null);

  useEffect(() => {
    fetch(`${assetUrl('pddca-viz-atlas/z_ranges.json')}?cb=${Z_CACHE}`)
      .then(r => r.json()).then(setData).catch(() => setData([]));
  }, []);

  if (!data) return <div className="static-page"><p style={{color:'#94a3b8'}}>Yükleniyor…</p></div>;

  return (
    <div className="static-page zr-page">
      <h2 className="static-page-title">Organ Z-Aralığı Analizi</h2>
      <p className="static-page-lead">
        {data.length} hasta × 9 organ. Her organın axial kesitler boyunca kapladığı Z aralığı,
        kesit sayısına göre normalize edilmiştir.
      </p>

      {/* Legend */}
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

      {/* View tabs */}
      <div className="zr-tabs">
        {[['gantt','Gantt (Z aralıkları)'],['span','Span Dağılımı'],['matrix','Eksik Organ Matrisi']].map(([id, label]) => (
          <button key={id} className={`zr-tab${view===id?' zr-tab-on':''}`} onClick={() => setView(id)}>{label}</button>
        ))}
      </div>

      {view === 'gantt'   && <ZRangeGantt   data={data} hoveredOrgan={hoveredOrgan} onHover={setHoveredOrgan}/>}
      {view === 'span'    && <ZRangeSpan    data={data} hoveredOrgan={hoveredOrgan} onHover={setHoveredOrgan}/>}
      {view === 'matrix'  && <ZRangeMatrix  data={data}/>}
    </div>
  );
}

function ZRangeGantt({ data, hoveredOrgan, onHover }) {
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
                    title={`${organ}: z ${o.z_min}–${o.z_max} (${o.span} kesit)`}
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
      <p className="zr-note">X ekseni: Z kesiti / toplam kesit sayısı (normalize). Her satır bir hasta.</p>
    </div>
  );
}

function ZRangeSpan({ data, hoveredOrgan, onHover }) {
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
              {/* Range bar */}
              <div className="zr-span-range" style={{left:`${pct(s.min)}%`, width:`${pct(s.max-s.min)}%`, background:`${col}33`}}/>
              {/* IQR box */}
              <div className="zr-span-iqr"  style={{left:`${pct(s.q1)}%`,  width:`${pct(s.q3-s.q1)}%`,   background:col, opacity:.7}}/>
              {/* Median line */}
              <div className="zr-span-med"  style={{left:`${pct(s.median)}%`, background:col}}/>
            </div>
            <div className="zr-span-nums">
              <span>min <b>{s.min}</b></span>
              <span>ort <b>{s.mean}</b></span>
              <span>max <b>{s.max}</b></span>
            </div>
          </div>
        );
      })}
      <div className="zr-span-axis">
        {[0,10,20,30,40,50].map(v => (
          <span key={v} style={{left:`${(v/maxSpan*100).toFixed(1)}%`}}>{v}</span>
        ))}
      </div>
      <p className="zr-note">Kutu: Q1–Q3 arası. Çizgi: medyan. Geniş alan: min–max. X: kesit sayısı.</p>
    </div>
  );
}

function ZRangeMatrix({ data }) {
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
                title={present ? `${organ}: z ${o.z_min}–${o.z_max}` : `${organ}: yok`}
                className="zr-matrix-cell"
                style={{background: present ? ORGAN_HEX[organ] : '#1e293b', opacity: present ? 0.85 : 1}}>
                {!present && <span className="zr-matrix-x">✕</span>}
              </span>
            );
          })}
        </div>
      ))}
      <p className="zr-note">Renkli = organ mevcut. ✕ = etiketsiz hasta.</p>
    </div>
  );
}

/* ── 2D Slice viewer ─────────────────────────────────────── */

function SliceViewerToggle({ src, patientId }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="slice-viewer-wrap">
      <button
        className={`slice-viewer-btn${open ? ' slice-viewer-btn-open' : ''}`}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="slice-viewer-btn-icon">{open ? '▲' : '▼'}</span>
        2D Kesit Görüntüleyici
        <span className="slice-viewer-btn-sub">
          {src ? 'HU · Ground Truth · Tahmin' : 'Sahne üretiliyor…'}
        </span>
      </button>

      {open && (
        <div className="slice-viewer-panel">
          {src ? (
            <iframe
              key={patientId}
              title={`2D Kesit — ${patientId}`}
              src={src}
              className="slice-viewer-iframe"
              loading="lazy"
            />
          ) : (
            <div className="slice-viewer-pending">
              <p>2D kesit sahnesi henüz üretilmedi.</p>
              <code>.venv/bin/python "2- uNET/export_slice_viewer.py" --pddca-root .</code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Static pages ────────────────────────────────────────── */

function DatasetPage() {
  return (
    <div className="static-page">
      <h2 className="static-page-title">Veri Seti: PDDCA</h2>
      <p className="static-page-lead">
        <strong>Public Domain Database for Computational Anatomy (PDDCA)</strong>, baş-boyun
        bölgesinin BT görüntülerini ve uzman segmentasyon etiketlerini içeren açık erişimli bir
        tıbbi görüntüleme veri setidir.
      </p>

      <div className="static-section">
        <h3 className="static-section-title">Genel Bilgiler</h3>
        <table className="static-table">
          <tbody>
            <tr><td>Modalite</td><td>Bilgisayarlı Tomografi (BT)</td></tr>
            <tr><td>Toplam hasta</td><td>48 (eğitim + test)</td></tr>
            <tr><td>Bu çalışmada test (off-site)</td><td>10 hasta</td></tr>
            <tr><td>Görüntü boyutu</td><td>~512 × 512 × 100–200 kesit</td></tr>
            <tr><td>Voksel aralığı</td><td>~0.98 × 0.98 × 3.0 mm</td></tr>
            <tr><td>Lisans</td><td>Kamuya açık (Public Domain)</td></tr>
          </tbody>
        </table>
      </div>

      <div className="static-section">
        <h3 className="static-section-title">Etiketlenen Yapılar (9 organ)</h3>
        <ul className="static-organ-list">
          {[
            ['Sağ parotis bezi', '#e74c3c'],
            ['Sol parotis bezi', '#e67e22'],
            ['Sağ submandibular bez', '#f1c40f'],
            ['Sol submandibular bez', '#2ecc71'],
            ['Sağ optik sinir', '#1abc9c'],
            ['Sol optik sinir', '#3498db'],
            ['Beyin sapı (brainstem)', '#9b59b6'],
            ['Mandibula (çene kemiği)', '#34495e'],
            ['Chiasma (optik kiyazma)', '#e91e63'],
          ].map(([name, color]) => (
            <li key={name} className="static-organ-item">
              <span className="static-organ-dot" style={{ background: color }} />
              {name}
            </li>
          ))}
        </ul>
      </div>

      <div className="static-section">
        <h3 className="static-section-title">Veri Bölme (Bu Çalışma)</h3>
        <table className="static-table">
          <thead><tr><th>Küme</th><th>Hasta sayısı</th><th>Kullanım</th></tr></thead>
          <tbody>
            <tr><td>Eğitim (train)</td><td>25</td><td>Model eğitimi</td></tr>
            <tr><td>Doğrulama (val)</td><td>5</td><td>Hiperparametre ayarı</td></tr>
            <tr><td>Test (in-site)</td><td>8</td><td>Ara değerlendirme</td></tr>
            <tr><td>Test (off-site)</td><td>10</td><td>Son değerlendirme — bu sitede gösterilen</td></tr>
          </tbody>
        </table>
      </div>

      <div className="static-section">
        <h3 className="static-section-title">Kaynak</h3>
        <p>
          Raudaschl P.F. et al., "Evaluation of segmentation methods on head and neck CT: Auto‑segmentation
          challenge 2015," <em>Medical Physics</em>, 2017.{' '}
          <a
            href="https://www.imagenglab.com/newsite/pddca/"
            target="_blank"
            rel="noreferrer"
            className="static-link"
          >
            imagenglab.com/newsite/pddca
          </a>
        </p>
      </div>
    </div>
  );
}

function StudiesPage() {
  return (
    <div className="static-page">
      <h2 className="static-page-title">Yapılan Çalışmalar</h2>
      <p className="static-page-lead">
        PDDCA veri seti üzerinde iki farklı otomatik segmentasyon yaklaşımı uygulanmış ve karşılaştırılmıştır.
      </p>

      <div className="static-section">
        <div className="study-card">
          <div className="study-card-header study-card-header-unet">
            <span className="study-card-icon">🧠</span>
            <h3 className="study-card-title">U-Net (Derin Öğrenme)</h3>
          </div>
          <div className="study-card-body">
            <p>
              2D U-Net mimarisi, 25 boyutlu giriş (merkez kesit ± 12 komşu kesit) ve
              64 × 64 ROI kırpma ile eğitilmiştir. Her organ için ayrı bir ikili
              segmentasyon başlığı kullanılmıştır.
            </p>
            <table className="static-table">
              <tbody>
                <tr><td>Mimari</td><td>2.5D U-Net (25-slice input)</td></tr>
                <tr><td>ROI boyutu</td><td>64 × 64 voksel</td></tr>
                <tr><td>Kayıp fonksiyonu</td><td>Dice + Binary Cross-Entropy</td></tr>
                <tr><td>Optimizer</td><td>Adam</td></tr>
                <tr><td>Framework</td><td>PyTorch</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="static-section">
        <div className="study-card">
          <div className="study-card-header study-card-header-atlas">
            <span className="study-card-icon">🗺️</span>
            <h3 className="study-card-title">Atlas Tabanlı Segmentasyon</h3>
          </div>
          <div className="study-card-body">
            <p>
              En yüksek kaliteli 5 atlas (eğitim hastası), SimpleITK ile her test hastasına
              kayıt edilmiş; kayıtlı etiketler üç farklı birleştirme stratejisiyle
              füze edilmiştir.
            </p>
            <table className="static-table">
              <tbody>
                <tr><td>Kayıt kütüphanesi</td><td>SimpleITK</td></tr>
                <tr><td>Kayıt türü</td><td>Afin + Deformable (Demons)</td></tr>
                <tr><td>Atlas sayısı</td><td>5 (en yüksek ortalama Dice)</td></tr>
              </tbody>
            </table>
            <h4 className="study-method-subtitle">Birleştirme yöntemleri</h4>
            <ul className="study-method-list">
              <li>
                <strong>HQ Deformable</strong> — En iyi tek atlas; doğrudan deformable
                kayıt çıktısı kullanılır.
              </li>
              <li>
                <strong>HQ Deformable + Majority Voting</strong> — 5 atlasın kayıtlı
                etiketleri voksel bazında çoğunluk oylamasıyla birleştirilir.
              </li>
              <li>
                <strong>HQ Deformable + STAPLE</strong> — Simultaneous Truth and
                Performance Level Estimation; her atlasın güvenilirliğini
                olasılıksal olarak modelleyen EM tabanlı füzyon.
              </li>
            </ul>
          </div>
        </div>
      </div>

      <div className="static-section">
        <h3 className="static-section-title">Değerlendirme Metriği</h3>
        <p>
          Segmentasyon kalitesi <strong>Dice Similarity Coefficient (DSC)</strong> ile
          ölçülmüştür. DSC = 2 × |A ∩ B| / (|A| + |B|), burada A tahmini, B ise
          ground truth segmentasyonudur. Değer aralığı: 0 (örtüşme yok) — 1 (mükemmel örtüşme).
        </p>
      </div>
    </div>
  );
}

function ContactPage() {
  return (
    <div className="static-page">
      <h2 className="static-page-title">İletişim</h2>
      <p className="static-page-lead">
        Bu çalışma hakkında sorularınız için aşağıdaki kanallardan ulaşabilirsiniz.
      </p>

      <div className="static-section">
        <div className="contact-card">
          <div className="contact-avatar">OC</div>
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
                <a
                  href="https://www.linkedin.com/in/olcay-%C3%A7oban-57084416b/"
                  target="_blank"
                  rel="noreferrer"
                  className="static-link"
                >
                  linkedin.com/in/olcay-çoban
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState('viewer');
  const [manifest, setManifest] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [selectedOrgan, setSelectedOrgan] = useState(null);
  const [atlasManifest, setAtlasManifest] = useState(null);
  const [atlasDiceDoc, setAtlasDiceDoc] = useState(null);
  const [atlasPairwiseDoc, setAtlasPairwiseDoc] = useState(null);
  const [atlasMetaDoc, setAtlasMetaDoc] = useState(null);
  const [selectedItems, setSelectedItems] = useState(['gt', 'unet', 'deformable', 'majority_voting', 'staple']);

  function toggleViewerItem(id) {
    setSelectedItems((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  }

  function getSceneUrl(id, patientId, manifest, atlasManifest) {
    const p = manifest?.patients?.find((pt) => pt.id === patientId);
    if (id === 'gt') {
      return p?.gt3d ? `${assetUrl(p.gt3d)}?cb=${SCENE3D_CACHE_BUST}` : null;
    }
    if (id === 'unet') {
      return p?.pred3d ? `${assetUrl(p.pred3d)}?cb=${SCENE3D_CACHE_BUST}` : null;
    }
    if (id === 'unet_fn') {
      return p?.fn3d ? `${assetUrl(p.fn3d)}?cb=${SCENE3D_CACHE_BUST}` : null;
    }
    if (id === 'unet_fp') {
      return p?.fp3d ? `${assetUrl(p.fp3d)}?cb=${SCENE3D_CACHE_BUST}` : null;
    }
    const m = atlasManifest?.methods?.find((m) => m.id === id);
    const rel = m?.scenes?.[patientId];
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
        if (data.patients?.length) {
          setPatientId((prev) => prev || data.patients[0].id);
        }
      } catch (e) {
        if (!cancelled) setLoadError(String(e.message || e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(
          `${assetUrl('pddca-viz-atlas/manifest.json')}?cb=${ATLAS_MANIFEST_CACHE_BUST}`
        );
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setAtlasManifest(data);
      } catch {
        if (!cancelled) setAtlasManifest(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(
          `${assetUrl('pddca-viz-atlas/atlas_dice_by_method.json')}?cb=${ATLAS_DICE_CACHE_BUST}`
        );
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setAtlasDiceDoc(data);
      } catch {
        if (!cancelled) setAtlasDiceDoc(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(
          `${assetUrl('pddca-viz-atlas/atlas_deformable_pairwise.json')}?cb=${ATLAS_PAIRWISE_CACHE_BUST}`
        );
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setAtlasPairwiseDoc(data);
      } catch {
        if (!cancelled) setAtlasPairwiseDoc(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(
          `${assetUrl('pddca-viz-atlas/atlas_meta.json')}?cb=${ATLAS_META_CACHE_BUST}`
        );
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setAtlasMetaDoc(data);
      } catch {
        if (!cancelled) setAtlasMetaDoc(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const current = useMemo(
    () => manifest?.patients?.find((p) => p.id === patientId),
    [manifest, patientId]
  );

  useEffect(() => {
    setSelectedOrgan(null);
  }, [patientId]);

  const NAV_ITEMS = [
    { id: 'viewer',   label: '3D Görselleştirme' },
    { id: 'zranges',  label: 'Organ Z-Aralıkları' },
    { id: 'dataset',  label: 'Veri Seti' },
    { id: 'studies',  label: 'Çalışmalar' },
    { id: 'contact',  label: 'İletişim' },
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="header-top">
          <h1>PDDCA — Baş-Boyun Organ Segmentasyonu</h1>
          <p className="header-sub">2.5D U-Net ve Atlas tabanlı yöntemlerin karşılaştırmalı değerlendirmesi</p>
        </div>
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
      </header>

      {page === 'zranges' && <ZRangePage />}
      {page === 'dataset' && <DatasetPage />}
      {page === 'studies' && <StudiesPage />}
      {page === 'contact' && <ContactPage />}

      {page === 'viewer' && loadError && (
        <div className="banner banner-error">
          manifest.json yüklenemedi: {loadError}
        </div>
      )}

      {page === 'viewer' && manifest && (
        <div className="app-body">
          {/* Sol sidebar */}
          <aside className="viewer-sidebar">
            <div className="viewer-sidebar-section">
              <label className="viewer-sidebar-label" htmlFor="patient-select">Hasta</label>
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
              <p className="viewer-sidebar-label">Gösterilecek görünümler</p>
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
                          {isReady ? 'hazır' : 'bekliyor'}
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
            </div>

            <div className="viewer-sidebar-divider" />
            <p className="viewer-sidebar-note">
              Sahneler arka planda üretiliyor. Sayfayı yenileyince yeni tamamlananlar görünür.
            </p>
          </aside>

          {/* Sağ ana alan */}
          <div className="app-main">
            {/* 2D Kesit görüntüleyici — 3D grid üstünde */}
            {current && (
              <section className="slice-viewer-section">
                <SliceViewerToggle
                  src={current.sliceViewer
                    ? `${assetUrl(current.sliceViewer)}?cb=${MANIFEST_CACHE_BUST}`
                    : null}
                  patientId={current.id}
                />
              </section>
            )}

            {/* 3B görüntüleyici grid */}
            {current && selectedItems.length > 0 && (
              <section className="viewer-grid-section">
                <div
                  className="viewer-grid"
                  style={{ '--vcol': Math.min(selectedItems.length, 2) }}
                >
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
                            <p className="atlas-3d-generate-title">Sahne henüz üretilmedi</p>
                            <p className="atlas-3d-generate-note">
                              Arka planda üretiliyor veya export betiği çalıştırılmamış.
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
            )}

            {/* Dice tabloları */}
            {current && (
              <>
                <DicePanel
                  patientId={current.id}
                  structureNames={manifest.structureNames}
                  organMetrics={current.organMetrics}
                  dicePercent={current.dicePercent}
                  selectedOrgan={selectedOrgan}
                  onSelectOrgan={setSelectedOrgan}
                />

                {atlasDiceDoc ? (
                  <AtlasDiceTriplet
                    patientId={current.id}
                    diceDoc={atlasDiceDoc}
                    structureNames={manifest.structureNames}
                    pairwiseDoc={atlasPairwiseDoc}
                    atlasMeta={atlasMetaDoc}
                  />
                ) : null}
              </>
            )}
          </div>
        </div>
      )}

      {page === 'viewer' && <AtlasArtifactsFooter />}
    </div>
  );
}