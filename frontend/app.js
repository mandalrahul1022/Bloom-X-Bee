/* ========= SAFE BOOTSTRAP (handles blocked CDNs) ========= */
/* ========= SAFE BOOTSTRAP (handles blocked CDNs) ========= */
const USE_HISTORY_MOCK = new URLSearchParams(location.search).get('mock') === '1';

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = resolve;
    s.onerror = () => reject(new Error('Failed to load ' + src));
    document.head.appendChild(s);
  });
}
function loadCSS(href) {
  return new Promise((resolve, reject) => {
    const l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = href;
    l.onload = resolve;
    l.onerror = () => reject(new Error('Failed to load ' + href));
    document.head.appendChild(l);
  });
}

// Load Leaflet + Chart (ONLY ONCE), then start.
(async () => {
  try {
    if (!document.querySelector('link[href*="leaflet.css"]')) {
      await loadCSS('https://unpkg.com/leaflet@1.9.4/dist/leaflet.css');
    }
    if (!window.L) {
      await loadScript('https://unpkg.com/leaflet@1.9.4/dist/leaflet.js');
    }
    // IMPORTANT: ensure exactly one Chart is present
    if (!window.Chart) {
      await loadScript('https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js');
    }
    // nice dark defaults once
    if (!window.__chartDefaultsSet && window.Chart) {
      Chart.defaults.color = '#cfe0ff';
      Chart.defaults.borderColor = 'rgba(207,224,255,0.2)';
      window.__chartDefaultsSet = true;
    }
  } catch (e) {
    console.warn('CDN load issue:', e.message);
  }
  startApp();
})();


// Global diagnostics (fires before/after app starts)
(function attachErrorTrap() {
  window.addEventListener('error', (ev) => {
    const msg = `Script error: ${ev.message || '(no message)'} @ ${ev.filename || 'inline'}:${ev.lineno || '?'}:${ev.colno || '?'}`;
    const t = document.getElementById('testStatus');
    if (t) { t.textContent = 'Tests: runtime error'; t.classList.remove('badge-ok'); t.classList.add('badge-err'); t.title = msg; }
    const d = document.getElementById('diagnostics');
    if (d) d.innerHTML = `<div>⚠︎ ${msg}</div>`;
  }, true);

  window.addEventListener('unhandledrejection', (ev) => {
    const msg = `Unhandled rejection: ${ev.reason?.message || ev.reason || '(no reason)'}`;
    const t = document.getElementById('testStatus');
    if (t) { t.textContent = 'Tests: runtime error'; t.classList.remove('badge-ok'); t.classList.add('badge-err'); t.title = msg; }
    const d = document.getElementById('diagnostics');
    if (d) d.innerHTML += `<div>⚠︎ ${msg}</div>`;
  });
})();

// Load Leaflet + Chart if they were blocked or missing, then start.
(async () => {
  try {
    if (!document.querySelector('link[href*="leaflet.css"]')) {
      await loadCSS('https://unpkg.com/leaflet@1.9.4/dist/leaflet.css');
    }
    if (!window.L) {
      await loadScript('https://unpkg.com/leaflet@1.9.4/dist/leaflet.js');
    }
    if (!window.Chart) {
      await loadScript('https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js');
    }
    if (window.Chart) {
      // pleasant defaults for dark theme
      Chart.defaults.color = '#cfe0ff';
      Chart.defaults.borderColor = 'rgba(207,224,255,0.20)';
    }
  } catch (e) {
    console.warn('CDN load issue:', e.message);
  }
  startApp();
})();

/* ======================= YOUR APP ======================== */
function startApp() {
  // --- CONFIG ---------------------------------------------------------------
  const DATA_URL = 'phenology_10_places.json';
  const USE_PIECEWISE_WARP = true;
  let HIVES_PER_KM2 = 4;
  let SUPPRESS_SEL_CHANGE = false;

  // Calibrated bee model presets (illustrative defaults; tune as needed)
  const BEE_PRESETS = {
    'central_valley_us': { r: 0.035, m: 0.010, Kmax: 70000, Bopt: 30000, eta: 0.00005 },
    'great_plains_us':   { r: 0.032, m: 0.011, Kmax: 65000, Bopt: 28000, eta: 0.000045 },
    'tucson_az_us':      { r: 0.028, m: 0.013, Kmax: 50000, Bopt: 24000, eta: 0.000035 },
    '_latbands': [
      { min: -90, max: -10, params: { r: 0.030, m: 0.012, Kmax: 60000, Bopt: 25000, eta: 0.00004 } },
      { min: -10, max:  10, params: { r: 0.033, m: 0.012, Kmax: 62000, Bopt: 26000, eta: 0.000042 } },
      { min:  10, max:  35, params: { r: 0.035, m: 0.011, Kmax: 68000, Bopt: 28000, eta: 0.000047 } },
      { min:  35, max:  60, params: { r: 0.032, m: 0.011, Kmax: 65000, Bopt: 27000, eta: 0.000045 } },
      { min:  60, max:  90, params: { r: 0.029, m: 0.012, Kmax: 58000, Bopt: 24000, eta: 0.000038 } },
    ]
  };
  function paramsFor(p) {
    if (BEE_PRESETS[p.id]) return BEE_PRESETS[p.id];
    const band = BEE_PRESETS._latbands.find(b => p.lat >= b.min && p.lat < b.max);
    return band ? band.params : {};
  }

  // UTIL
  const fmtDate = (s) => { const d = new Date(s); return isNaN(d) ? String(s) : d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }); };
  const isoDay = (s) => { if (typeof s !== 'string') return null; const m = /^\d{4}-\d{2}-\d{2}/.exec(s.trim()); return m ? m[0] : null; };
  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  function indexToPixel(xScale, idx, labels) {
    const i = clamp(idx | 0, 0, labels.length - 1);
    const labelValue = labels[i];
    try { return xScale.getPixelForValue(labelValue); }
    catch (e) { try { return xScale.getPixelForValue(i); } catch (e2) { return NaN; } }
  }

  // FALLBACK DATA
  function synthesizeData() {
    const today = new Date();
    const start = new Date(today.getTime() - 365 * 24 * 3600 * 1000);
    const days = 365;
    function seasonal(dayOfYear, peak, width, amp, base) { const x = (dayOfYear - peak) / width; return Math.max(0, Math.min(1, base + amp * Math.exp(-0.5 * x * x))); }
    function series(peak, width, amp, base) {
      const out = [];
      for (let i = 0; i < days; i++) {
        const d = new Date(start.getTime() + i * 24 * 3600 * 1000);
        const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
        const doy = Math.floor((Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) - yearStart.getTime()) / 86400000) + 1;
        let v = seasonal(doy, peak, width, amp, base); v += (Math.random() * 0.04 - 0.02); v = Math.max(0, Math.min(1, v));
        out.push({ date: d.toISOString(), evi2: Number(v.toFixed(4)) });
      }
      return out;
    }
    function detect(se) { const ev = se.map(x => x.evi2); const vmin = Math.min(...ev), vmax = Math.max(...ev); const thr = vmin + 0.2 * (vmax - vmin); let sos = null, eos = null, peakIdx = ev.indexOf(vmax); ev.forEach((v, i) => { if (sos === null && v >= thr) sos = i; if (v >= thr) eos = i; }); return { sos, peakIdx, eos, vmin, vmax, range: Number((vmax - vmin).toFixed(4)) }; }
    const PLACES = [
      { id: 'central_valley_us', name: 'Central Valley, California, USA', lat: 36.6, lon: -119.6, p: { peak: 110, width: 30, amp: 0.45, base: 0.2 }, species: ['California poppy', 'Almond', 'Grape'], npn: true },
      { id: 'great_plains_us', name: 'Great Plains, USA', lat: 41.5, lon: -100.0, p: { peak: 180, width: 40, amp: 0.46, base: 0.18 }, species: ['Sunflower', 'Prairie clover', 'Bluestem'], npn: true },
      { id: 'tucson_az_us', name: 'Tucson, Arizona, USA', lat: 32.2217, lon: -110.9265, p: { peak: 95, width: 28, amp: 0.43, base: 0.16 }, species: ['Saguaro', 'Creosote bush', 'Mesquite'], npn: true },
      { id: 'gainesville_fl_us', name: 'Gainesville, Florida, USA', lat: 29.6516, lon: -82.3248, p: { peak: 140, width: 34, amp: 0.50, base: 0.24 }, species: ['Azalea', 'Saw palmetto', 'Live oak'], npn: true },
      { id: 'ithaca_ny_us', name: 'Ithaca, New York, USA', lat: 42.4439, lon: -76.5019, p: { peak: 155, width: 26, amp: 0.48, base: 0.19 }, species: ['Red maple', 'Apple', 'Lilac'], npn: true },
      { id: 'minneapolis_mn_us', name: 'Minneapolis, Minnesota, USA', lat: 44.9778, lon: -93.2650, p: { peak: 170, width: 28, amp: 0.46, base: 0.18 }, species: ['Prairie coneflower', 'Milkweed', 'Sugar maple'], npn: true },
      { id: 'lisse_netherlands', name: 'Lisse, Netherlands (Tulip Region)', lat: 52.26, lon: 4.55, p: { peak: 120, width: 22, amp: 0.50, base: 0.18 }, species: ['Tulip', 'Hyacinth', 'Daffodil'] },
      { id: 'kyoto_japan', name: 'Kyoto, Japan', lat: 35.01, lon: 135.77, p: { peak: 105, width: 24, amp: 0.48, base: 0.20 }, species: ['Cherry blossom', 'Camellia', 'Plum'] },
      { id: 'kathmandu_nepal', name: 'Kathmandu Valley, Nepal', lat: 27.72, lon: 85.32, p: { peak: 170, width: 36, amp: 0.52, base: 0.22 }, species: ['Rhododendron', 'Mustard', 'Marigold'] },
      { id: 'sao_paulo_brazil', name: 'São Paulo, Brazil', lat: -23.55, lon: -46.63, p: { peak: 310, width: 34, amp: 0.50, base: 0.25 }, species: ['Ipê amarelo', 'Coffee', 'Jacaranda'] }
    ];
    const places = PLACES.map(pl => {
      const se = series(pl.p.peak, pl.p.width, pl.p.amp, pl.p.base);
      const { sos, peakIdx, eos, vmin, vmax, range } = detect(se);
      return {
        id: pl.id, name: pl.name, lat: pl.lat, lon: pl.lon,
        sos_date: se[sos]?.date || se[0].date,
        peak_date: se[peakIdx]?.date || se[Math.floor(se.length / 2)].date,
        eos_date: se[eos]?.date || se[se.length - 1].date,
        series: se,
        highest_bloom: Number(vmax.toFixed(4)),
        lowest_bloom: Number(vmin.toFixed(4)),
        bloom_range: range,
        species: pl.species,
        notes: 'Synthetic EVI2-like curve (fallback).'
      };
    });
    return { generated_on: new Date().toISOString(), places };
  }

  // DATA LOAD
  async function loadData() {
    try {
      const resp = await fetch(DATA_URL, { cache: 'no-store' });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const json = await resp.json();
      if (!json || !Array.isArray(json.places)) throw new Error('Malformed JSON: missing places[]');
      return { json, source: 'link' };//how to add link
    } catch (e) {
      console.warn('Falling back to synthetic data:', e);
      return { json: synthesizeData(), source: 'embedded-synth' };
    }
  }

  // CHART PLUGINS (guarded so app never crashes if Chart is missing)
  const vlinePlugin = {
    id: 'vline',
    afterDraw(chart, args, opts) {
      const { ctx, chartArea, scales, data } = chart;
      const x = scales.x; const labels = data.labels || [];
      (opts.indices || []).forEach(it => {
        if (typeof it.idx !== 'number' || it.idx < 0 || !labels.length) return;
        const xPos = indexToPixel(x, it.idx, labels);
        if (!isFinite(xPos)) return;
        ctx.save();
        ctx.lineWidth = 1; ctx.setLineDash([6, 6]);
        ctx.strokeStyle = it.color || '#9cb0ff';
        ctx.beginPath(); ctx.moveTo(xPos, chartArea.top); ctx.lineTo(xPos, chartArea.bottom); ctx.stroke();
        ctx.restore();
      });
    }
  };
  const xbandsPlugin = {
    id: 'xbands',
    beforeDatasetsDraw(chart, args, opts) {
      const bands = (opts && Array.isArray(opts.bands)) ? opts.bands : [];
      if (!bands.length) return;
      const { ctx, chartArea, scales, data } = chart; const x = scales.x; const labels = data.labels || [];
      ctx.save();
      bands.forEach(b => {
        const x1 = indexToPixel(x, clamp(b.startIdx | 0, 0, labels.length - 1), labels);
        const x2 = indexToPixel(x, clamp(b.endIdx | 0, 0, labels.length - 1), labels);
        if (!isFinite(x1) || !isFinite(x2)) return;
        const left = Math.min(x1, x2), right = Math.max(x1, x2);
        ctx.fillStyle = b.fill || 'rgba(255, 209, 102, 0.12)';
        ctx.fillRect(left, chartArea.top, right - left, chartArea.bottom - chartArea.top);
      });
      ctx.restore();
    }
  };
 if (window.Chart && typeof Chart.register === 'function' && !window.__pluginsRegistered) {
  Chart.register(vlinePlugin, xbandsPlugin);
  window.__pluginsRegistered = true;
} else if (!window.Chart) {
  console.warn('Chart.js not available; charts disabled.');
}

  // STATE
  let map, markers = []; let chart, dataAll; let chartCtx; let beeChart, beeCtx; let trend;

  function mkChart(labels, values, idx) {
    if (!window.Chart) return;
    if (chart) chart.destroy();
    chart = new Chart(chartCtx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'EVI2 (this year)', data: values, tension: 0.25, pointRadius: 0, borderWidth: 2 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { x: { ticks: { maxTicksLimit: 6 } }, y: { min: 0, max: 1 } },
        plugins: { legend: { display: true }, vline: { indices: idx }, xbands: { bands: [] } }
      }
    });
  }
  function setUncertaintyBands(bands) {
    if (!chart) return;
    chart.options.plugins.xbands = chart.options.plugins.xbands || {};
    chart.options.plugins.xbands.bands = bands || [];
    chart.update();
  }
  function addForecastToChart(forecastData) {
    if (!chart || !forecastData || !window.Chart) return;
    chart.data.datasets = chart.data.datasets.filter(ds => ds._isForecast !== true);
    chart.data.datasets.push({
      label: 'Projected next year',
      data: forecastData, tension: 0.25, pointRadius: 0, borderWidth: 2,
      borderDash: [6, 4], borderColor: '#ffd166', backgroundColor: 'rgba(255, 209, 102, 0.15)', _isForecast: true
    });
    chart.update();
  }
  function mkBeeChart(labels, bees, honey) {
    if (!window.Chart) return;
    if (beeChart) beeChart.destroy();
    beeChart = new Chart(beeCtx, {
      type: 'line',
      data: {
        labels, datasets: [
          { label: 'Colony size (bees)', data: bees, yAxisID: 'y1', tension: 0.2, pointRadius: 0, borderWidth: 2 },
          { label: 'Honey/day (kg)', data: honey, yAxisID: 'y2', tension: 0.2, pointRadius: 0, borderWidth: 2 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { ticks: { maxTicksLimit: 6 } },
          y1: { type: 'linear', position: 'left', beginAtZero: true },
          y2: { type: 'linear', position: 'right', beginAtZero: true, grid: { drawOnChartArea: false } }
        },
        plugins: { legend: { display: true } }
      }
    });
  }

  // Models & Helpers
  function activityFromLatDay(lat, doy) { const season = 0.6 + 0.4 * Math.cos(2 * Math.PI * (doy - 200) / 365) * Math.min(1, Math.abs(lat) / 45); return Math.max(0, Math.min(1, season)); }
  function computeForageSeries(p) {
    const labels = p.series.map(d => d.date);
    const evi = p.series.map(d => d.evi2);
    const speciesBoost = p._inat_count ? (1 + 0.2 * Math.log(1 + p._inat_count)) : 1;
    const npnBoost = (typeof p._npn_pct === 'number') ? (1 + 0.5 * (p._npn_pct)) : 1;
    const f = evi.map(v => Math.max(0, Math.min(1, v * npnBoost * speciesBoost)));
    const act = labels.map((d) => { const day = new Date(d); const yearStart = new Date(Date.UTC(day.getUTCFullYear(), 0, 1)); const doy = Math.floor((Date.UTC(day.getUTCFullYear(), day.getUTCMonth(), day.getUTCDate()) - yearStart.getTime()) / 86400000) + 1; return activityFromLatDay(p.lat, doy); });
    const F = f.map((v, i) => Number((v * act[i]).toFixed(4))); return { labels, evi, F };
  }
  function daysBetween(aIso, bIso) { const a = new Date(aIso), b = new Date(bIso); return Math.round((a.getTime() - b.getTime()) / 86400000); }
  function shiftWrap(arr, k) { const n = arr.length; if (!n) return arr.slice(); const s = ((k % n) + n) % n; const out = new Array(n); for (let i = 0; i < n; i++) { out[(i + s) % n] = arr[i]; } return out; }
  function resampleLinear(arr, newLen) { if (newLen <= 1) return [arr[0]]; const n = arr.length; const out = new Array(newLen); for (let i = 0; i < newLen; i++) { const pos = i * (n - 1) / (newLen - 1); const j = Math.floor(pos); const t = pos - j; const a = arr[Math.min(j, n - 1)]; const b = arr[Math.min(j + 1, n - 1)]; out[i] = a * (1 - t) + b * t; } return out; }
  function piecewiseWarp(values, iS, iP, iE, predS, predP, predE) {
    const N = values.length; const base = Math.min(...values);
    if (!(iS >= 0 && iP > iS && iE > iP)) return shiftWrap(values, predS - iS);
    const seg1 = values.slice(iS, iP + 1); const seg2 = values.slice(iP, iE + 1);
    const len1 = Math.max(2, predP - predS); const len2 = Math.max(2, predE - predP);
    const r1 = resampleLinear(seg1, len1); const r2 = resampleLinear(seg2, len2);
    const proj = new Array(N).fill(base); let idx = ((predS % N) + N) % N;
    r1.concat(r2).forEach(v => { proj[idx] = Math.max(base, Math.min(1, v)); idx = (idx + 1) % N; });
    return proj;
  }
  function piecewiseWarpBlend(values, iS, iP, iE, predS, predP, predE) {
    const N = values.length; const base = shiftWrap(values, (predS - iS));
    if (!(iS >= 0 && iP > iS && iE > iP)) return base;
    const warped = piecewiseWarp(values, iS, iP, iE, predS, predP, predE);
    let idx = ((predS % N) + N) % N; const end = ((predE % N) + N) % N; let k = idx;
    while (true) { base[k] = Math.max(0, Math.min(1, warped[k])); if (k === end) break; k = (k + 1) % N; }
    return base;
  }
  function classifyStates(evi) {
    const n = evi.length; const states = new Array(n).fill(0);
    const der = evi.map((v, i) => i ? v - evi[i - 1] : 0);
    const hi = 0.6, lo = 0.25, up = 0.004, down = -0.004;
    for (let i = 0; i < n; i++) {
      const v = evi[i], d = der[i];
      if (v < lo) { states[i] = 0; continue; }
      if (v >= hi && Math.abs(d) < 0.002) { states[i] = 2; continue; }
      if (d >= up) { states[i] = 1; continue; }
      if (d <= down) { states[i] = 3; continue; }
      states[i] = v < lo ? 4 : 2;
    }
    for (let i = 0; i < n; i++) { if (evi[i] < 0.15) states[i] = 4; }
    return states;
  }
  function learnTransitions(states) {
    const K = 5; const M = Array.from({ length: K }, () => Array(K).fill(0));
    for (let i = 1; i < states.length; i++) { M[states[i - 1]][states[i]]++; }
    for (let r = 0; r < K; r++) { let s = M[r].reduce((a, b) => a + b, 0) + K; for (let c = 0; c < K; c++) M[r][c] = (M[r][c] + 1) / s; }
    return M;
  }
  function durationsByState(states) { const K = 5, d = Array(K).fill(0); states.forEach(s => d[s]++); return d; }
  function predictNextSeason(p) {
    const evi = p.series.map(d => d.evi2);
    const labels = p.series.map(d => new Date(d.date));
    const states = classifyStates(evi);
    const trans = learnTransitions(states);
    const durs = durationsByState(states);
    const total = states.length;
    const last = labels[labels.length - 1];
    const nextStart = new Date(last.getTime() + 24 * 3600 * 1000);
    const next = []; let s0 = states[states.length - 1]; let cur = new Date(nextStart);
    const argmax = (row) => row.indexOf(Math.max(...row));
    for (let i = 0; i < total; i++) { s0 = argmax(trans[s0]); next.push({ date: new Date(cur), state: s0 }); cur = new Date(cur.getTime() + 24 * 3600 * 1000); }
    const idxOf = (st) => next.findIndex(x => x.state === st);
    const idxSoS = idxOf(1);
    const idxPeak = idxOf(2);
    let idxEoS = -1;
    if (idxPeak >= 0) { idxEoS = next.slice(idxPeak).findIndex(x => x.state === 3); if (idxEoS >= 0) idxEoS += idxPeak; }
    const safe = (i) => (i >= 0 ? next[i].date : new Date(nextStart.getTime() + (durs[1] || 50) * 24 * 3600 * 1000));
    const sos = safe(idxSoS);
    const peak = safe(idxPeak >= 0 ? idxPeak : (idxSoS + (durs[1] || 40)));
    const eos = safe(idxEoS >= 0 ? idxEoS : (idxPeak + (durs[2] || 30)));
    const hiveLeadDays = 14;
    const hiveStart = new Date(sos.getTime() - hiveLeadDays * 24 * 3600 * 1000);
    return { sos: sos.toISOString(), peak: peak.toISOString(), eos: eos.toISOString(), hive_start: hiveStart.toISOString(), markov: { trans, durations: durs } };
  }

  function simulateBee(p, opts = {}) {
    const base = { r: 0.03, m: 0.012, Kmax: 60000, Bopt: 25000, eta: 0.00004, B0: 12000 };
    const tuned = Object.assign({}, base, paramsFor(p), opts);
    const { labels, F } = computeForageSeries(p);
    const n = F.length; const { r, m, Kmax, Bopt, eta, B0 } = tuned;
    let B = B0; const bees = [], honey = []; let good = 0, dearth = 0, dearthMax = 0;
    for (let i = 0; i < n; i++) {
      const Kt = Math.max(2000, Kmax * F[i]);
      const dB = r * B * (1 - B / Kt) - m * B;
      B = Math.max(0, B + dB);
      bees.push(Math.round(B));
      const dayHoney = eta * F[i] * Math.min(1, B / Bopt) * Bopt;
      honey.push(Number(dayHoney.toFixed(3)));
      if (F[i] > 0.4) { good++; dearth = 0; } else { dearth++; dearthMax = Math.max(dearthMax, dearth); }
    }
    const totalHoney = Number(honey.reduce((a, b) => a + b, 0).toFixed(1));
    return { labels, bees, honey, totalHoney, goodDays: good, dearthMax };
  }

  function setPlace(p) {
    if (!p || !Array.isArray(p.series)) return;
    p._npn_pct = undefined; p._inat_count = undefined;
    const sel = document.getElementById('placeSel');
    if (sel && sel.value !== p.id) { SUPPRESS_SEL_CHANGE = true; sel.value = p.id; SUPPRESS_SEL_CHANGE = false; }

    document.getElementById('where').textContent = p.name;
    document.getElementById('meta').innerHTML = `
      <span class="pill">Lat: ${Number(p.lat).toFixed(2)}</span>
      <span class="pill">Lon: ${Number(p.lon).toFixed(2)}</span>
      <span class="pill">SoS: ${fmtDate(p.sos_date)}</span>
      <span class="pill">Peak: ${fmtDate(p.peak_date)}</span>
      <span class="pill">EoS: ${fmtDate(p.eos_date)}</span>
    `;
    const labels = p.series.map(d => d.date);
    const values = p.series.map(d => d.evi2);
    const idxOf = (iso) => { const target = isoDay(iso); if (!target) return -1; return labels.findIndex(d => isoDay(d) === target); };
    const idx = [{ idx: idxOf(p.sos_date), color: '#7cf7d4' }, { idx: idxOf(p.peak_date), color: '#e7edff' }, { idx: idxOf(p.eos_date), color: '#9cb0ff' }];
    mkChart(labels, values, idx);
    if (window.L) map.setView([p.lat, p.lon], 5);

    const highest = p.highest_bloom ?? Math.max(...values);
    const lowest = p.lowest_bloom ?? Math.min(...values);
    const range = p.bloom_range ?? Number((highest - lowest).toFixed(4));
    const species = Array.isArray(p.species) ? p.species : [];
    document.getElementById('stats').innerHTML = `
      <div class="meta">
        <span class="pill">Highest bloom: ${highest.toFixed(4)}</span>
        <span class="pill">Lowest bloom: ${lowest.toFixed(4)}</span>
        <span class="pill">Bloom range: ${range.toFixed ? range.toFixed(4) : range}</span>
      </div>
      ${species.length ? '<div style="margin-top:8px">Species Observed:<ul>' + species.map(s => `<li>${s}</li>`).join('') + '</ul></div>' : ''}
    `;

    document.getElementById('npn').innerHTML = p.npn ? '<span class="pill">NPN: fetching open flowers…</span>' : '';
    document.getElementById('inat').textContent = '';
    if (p.npn) { loadNPN(p).catch(() => { document.getElementById('npn').innerHTML = '<span class="pill">NPN: unavailable</span>'; }); }
    loadINat(p).catch(() => { });
    updateFeasibility(p);
    loadHistoryTrend(p, 6).catch(() => { });
  }

  // Map & UI
  function addMarkers(places) {
    if (!window.L) return;
    markers.forEach(m => m.remove()); markers = [];
    places.forEach(p => {
      const m = L.marker([p.lat, p.lon]).addTo(map)
        .bindPopup(`<b>${p.name}</b><br>SoS: ${fmtDate(p.sos_date)}<br>Peak: ${fmtDate(p.peak_date)}<br>EoS: ${fmtDate(p.eos_date)}`);
      m.on('click', () => setPlace(p)); markers.push(m);
    });
  }
  function populateSelect(places) {
    const sel = document.getElementById('placeSel');
    sel.innerHTML = '';
    const ph = document.createElement('option'); ph.value = ''; ph.disabled = true; ph.selected = true; ph.textContent = '⟡ Select a place…'; sel.appendChild(ph);
    places.forEach(p => { const opt = document.createElement('option'); opt.value = p.id; opt.textContent = p.name; sel.appendChild(opt); });
    sel.addEventListener('change', () => { if (SUPPRESS_SEL_CHANGE) return; const p = places.find(x => x.id === sel.value); if (p) setPlace(p); });
  }

  // Best start window + forecast bands
  function updateFeasibility(p) {
    const sim = simulateBee(p, {});
    mkBeeChart(sim.labels, sim.bees, sim.honey);
    const feasEl = document.getElementById('feas');
    const honeyPerColony = sim.totalHoney;
    const perKm2 = Math.round(honeyPerColony * HIVES_PER_KM2);
    const perAcre = Math.round(perKm2 / 2.59);
    const risk = HIVES_PER_KM2 > 8 ? '⚠︎ possible overstocking' : '';
    feasEl.innerHTML = `<div class="meta"><span class="pill">Honey/colony (kg): ${honeyPerColony}</span><span class="pill">Honey per km² (kg): ${perKm2}</span><span class="pill">Honey per acre (kg): ${perAcre}</span>${risk ? `<span class="pill">${risk}</span>` : ''}</div>`;

    const fc = predictNextSeason(p);
    const labels = p.series.map(d => d.date); const values = p.series.map(d => d.evi2);
    const idxOf = (iso) => { const target = isoDay(iso); if (!target) return -1; return labels.findIndex(d => isoDay(d) === target); };
    const iS = idxOf(p.sos_date), iP = idxOf(p.peak_date), iE = idxOf(p.eos_date);
    let projected;
    if (USE_PIECEWISE_WARP) {
      const baseDate0 = new Date(labels[0]);
      const toIdx = (iso) => Math.max(0, Math.min(values.length - 1, Math.round((new Date(iso) - baseDate0) / 86400000)));
      const pS = toIdx(fc.sos), pP = toIdx(fc.peak), pE = toIdx(fc.eos);
      projected = piecewiseWarpBlend(values, iS, iP, iE, pS, pP, pE);
    } else {
      const deltaPeak = daysBetween(fc.peak, p.peak_date);
      projected = shiftWrap(values, deltaPeak);
    }
    projected = projected.map(v => (isFinite(v) ? Math.max(0, Math.min(1, v)) : 0));
    addForecastToChart(projected);

    const unc = p.uncertainty_days || { sos: 7, peak: 6, eos: 8 };
    const baseDate0 = new Date(labels[0]); const toIdx = (iso) => Math.max(0, Math.min(values.length - 1, Math.round((new Date(iso) - baseDate0) / 86400000)));
    const sosIdx = toIdx(fc.sos), peakIdx = toIdx(fc.peak), eosIdx = toIdx(fc.eos);
    const bands = [
      { startIdx: sosIdx - Math.round(unc.sos || 7), endIdx: sosIdx + Math.round(unc.sos || 7), fill: 'rgba(124,247,212,0.10)' },
      { startIdx: peakIdx - Math.round(unc.peak || 6), endIdx: peakIdx + Math.round(unc.peak || 6), fill: 'rgba(255,209,102,0.12)' },
      { startIdx: eosIdx - Math.round(unc.eos || 8), endIdx: eosIdx + Math.round(unc.eos || 8), fill: 'rgba(156,176,255,0.10)' }
    ];
    setUncertaintyBands(bands);

    // Decision window: best time to start hives next season
    const hiveLeadDays = 14;
    const sosSigma = Math.round(unc.sos || 7);
    const startMid = new Date(new Date(fc.sos).getTime() - hiveLeadDays * 86400000);
    const startEarly = new Date(startMid.getTime() - sosSigma * 86400000);
    const startLate = new Date(startMid.getTime() + sosSigma * 86400000);
    const forecastEl = document.getElementById('forecast');
    forecastEl.innerHTML = `<div class="meta">
      <span class="pill">Pred SoS: ${fmtDate(fc.sos)}</span>
      <span class="pill">Pred Peak: ${fmtDate(fc.peak)}</span>
      <span class="pill">Pred EoS: ${fmtDate(fc.eos)}</span>
      <span class="pill">Best start window: ${fmtDate(startEarly)} – ${fmtDate(startLate)}</span>
      <span class="pill">(midpoint: ${fmtDate(startMid.toISOString())})</span>
    </div>`;
  }

  // Live lookups
  function windowFor(p) { const s = isoDay(p.sos_date), e = isoDay(p.eos_date), peak = isoDay(p.peak_date); if (s && e) return { d1: s, d2: e }; const pd = new Date(peak || p.series[Math.floor(p.series.length / 2)].date); const d1 = new Date(pd.getTime() - 30 * 86400000).toISOString().slice(0, 10); const d2 = new Date(pd.getTime() + 30 * 86400000).toISOString().slice(0, 10); return { d1, d2 }; }
  async function loadNPN(p) { const { d1, d2 } = windowFor(p); const url = `/api/npn_openflowers?lat=${p.lat}&lon=${p.lon}&from=${d1}&to=${d2}&requestSource=${encodeURIComponent('Caldwell University')}&contact=${encodeURIComponent('Rahul Mandal')}`; const r = await fetch(url, { cache: 'no-store' }); if (!r.ok) throw new Error('npn http ' + r.status); const j = await r.json(); const pct = typeof j.percent === 'number' ? j.percent : (j.total ? (100 * j.open / j.total) : 0); p._npn_pct = (pct / 100); document.getElementById('npn').innerHTML = `<span class="pill">NPN open flowers: ${pct.toFixed(1)}% (${j.open ?? '?'} / ${j.total ?? '?'})</span>`; updateFeasibility(p); }
  async function loadINat(p) { const { d1, d2 } = windowFor(p); const params = new URLSearchParams({ taxon_id: '47126', lat: String(p.lat), lng: String(p.lon), radius: '25', d1, d2, quality_grade: 'research', per_page: '200', annotation_term_id: '12', annotation_value_id: '13' }); try { const r = await fetch(`https://api.inaturalist.org/v1/observations?${params.toString()}`); if (!r.ok) throw new Error('inat http ' + r.status); const j = await r.json(); const seen = new Set(); (j.results || []).forEach(o => { const n = o?.taxon?.name; if (n) seen.add(n); }); p._inat_count = seen.size; document.getElementById('inat').innerHTML = `<div>Observed near peak (iNat): <b>${seen.size}</b> species</div>`; } catch (e) { /* silent fail */ } updateFeasibility(p); }

  // Multi-year trend loader
  async function loadHistoryTrend(p, years = 6) {
    const url = `/api/evi2_history?lat=${p.lat}&lon=${p.lon}&years=${years}` + (USE_HISTORY_MOCK ? '&mock=1' : '');
    try {
      const r = await fetch(url, { cache: 'no-store' });
      if (!r.ok) throw new Error(`history HTTP ${r.status}`);
      const j = await r.json();

      const yearsArr = (j.records || []).map(r => r.year);
      const fix = a => (a || []).map(v => (v == null ? undefined : v));
      const sos = fix(j.records.map(r => r.sos_doy));
      const peak = fix(j.records.map(r => r.peak_doy));
      const eos = fix(j.records.map(r => r.eos_doy));

      const ctx = document.getElementById('trendChart')?.getContext('2d');
      if (!ctx || !window.Chart) return;

      if (trend) trend.destroy();
      trend = new Chart(ctx, {
        type: 'line',
        data: {
          labels: yearsArr,
          datasets: [
            { label: 'SoS (DoY)',  data: sos,  pointRadius: 3, borderWidth: 2, spanGaps: true, tension: 0.25,
              borderColor: '#7CF7D4', backgroundColor: 'rgba(124,247,212,0.15)' },
            { label: 'Peak (DoY)', data: peak, pointRadius: 3, borderWidth: 2, spanGaps: true, tension: 0.25,
              borderColor: '#FFD166', backgroundColor: 'rgba(255,209,102,0.15)' },
            { label: 'EoS (DoY)',  data: eos,  pointRadius: 3, borderWidth: 2, spanGaps: true, tension: 0.25,
              borderColor: '#9CB0FF', backgroundColor: 'rgba(156,176,255,0.15)' },
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { min: 0, max: 366, ticks: { stepSize: 30 }, grid: { color: 'rgba(255,255,255,0.08)' } },
            x: { grid: { color: 'rgba(255,255,255,0.05)' } }
          },
          plugins: { legend: { display: true } }
        }
      });
    } catch (err) {
      console.error('history trend failed:', err);
      const diag = document.getElementById('diagnostics');
      if (diag) diag.innerHTML += `<div>Trend error: ${String(err.message || err)}</div>`;
    }
  }

  // MAIN
  (async function main() {
    const chartCanvas = document.getElementById('chart');
    const beeCanvas = document.getElementById('beeChart');
    if (!chartCanvas || !beeCanvas) {
      const d = document.getElementById('diagnostics'); if (d) { d.textContent = 'Critical: canvases missing'; }
      return;
    }
    chartCtx = chartCanvas.getContext('2d'); beeCtx = beeCanvas.getContext('2d');

    const { json, source } = await loadData(); dataAll = json;
    const dsEl = document.getElementById('dataSource'); if (dsEl) dsEl.textContent = `Source: ${source}`;

    if (!window.L) { console.error('Leaflet not available'); return; }
    map = L.map('map', { zoomControl: true, worldCopyJump: true }).setView([20, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 8, attribution: '© OpenStreetMap' }).addTo(map);

    if (!dataAll || !Array.isArray(dataAll.places) || dataAll.places.length !== 10) {
      const d = document.getElementById('diagnostics'); if (d) { d.textContent = 'Data problem: using synthetic fallback.'; }
      dataAll = synthesizeData();
    }

    addMarkers(dataAll.places); populateSelect(dataAll.places);
    // keep placeholder until user selects

    const hp = document.getElementById('hivesPerKm2');
    if (hp) {
      hp.addEventListener('change', () => {
        const v = Number(hp.value); HIVES_PER_KM2 = (isFinite(v) && v > 0) ? v : 4;
        const sel = document.getElementById('placeSel');
        const p = dataAll.places.find(x => x.id === sel.value);
        if (p) updateFeasibility(p);
      });
    }
  })();
} // end startApp()
