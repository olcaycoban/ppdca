import { useEffect, useMemo, useState } from 'react';
import './App.css';

const publicBase = process.env.PUBLIC_URL || '';
/** Tarayıcı eski scene_3d.html önbelleğinden vermesin (export sonrası yenile). */
const SCENE3D_CACHE_BUST = 'v6';
const MANIFEST_CACHE_BUST = 'v10';

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

function Viz3dFrame({ src, title }) {
  return (
    <figure className="viz-figure viz-figure-3d">
      <div className="viz-frame viz-frame-3d">
        <iframe
          title={title}
          className="viz-iframe-3d"
          src={src}
          loading="lazy"
        />
      </div>
      <figcaption className="viz-3d-caption">
        Üst: GT ve tahmin. Alt satır: uyuşmazlıklar (yüzey veya ince bölgelerde nokta bulutu). Export sonrası
        hâlâ boşsa Cmd+Shift+R ile sert yenileyin.
      </figcaption>
    </figure>
  );
}

export default function App() {
  const [manifest, setManifest] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [selectedOrgan, setSelectedOrgan] = useState(null);

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

  const current = useMemo(
    () => manifest?.patients?.find((p) => p.id === patientId),
    [manifest, patientId]
  );

  useEffect(() => {
    setSelectedOrgan(null);
  }, [patientId]);

  return (
    <div className="app">
      <header className="header">
        <h1>PDDCA — 3D karşılaştırma</h1>
        <p className="header-sub">Test off-site 10 hasta · Fareyle döndürün, tekerlekle yakınlaştırın.</p>
      </header>

      {loadError && (
        <div className="banner banner-error">
          manifest.json yüklenemedi: {loadError}
        </div>
      )}

      {manifest && (
        <>
          <div className="toolbar">
            <label htmlFor="patient">Hasta </label>
            <select
              id="patient"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="select-patient"
            >
              {manifest.patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.id}
                </option>
              ))}
            </select>
          </div>

          {current && (
            <main className="main main-3d-only">
              {current.scene3d ? (
                <Viz3dFrame
                  key={`${current.id}-3d`}
                  src={`${assetUrl(current.scene3d)}?cb=${SCENE3D_CACHE_BUST}`}
                  title={`3D ${current.id}`}
                />
              ) : null}

              <DicePanel
                patientId={current.id}
                structureNames={manifest.structureNames}
                organMetrics={current.organMetrics}
                dicePercent={current.dicePercent}
                selectedOrgan={selectedOrgan}
                onSelectOrgan={setSelectedOrgan}
              />
            </main>
          )}
        </>
      )}
    </div>
  );
}
