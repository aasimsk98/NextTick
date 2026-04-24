/* =================================================================
   NextTick — frontend controller
   ================================================================= */

(() => {
  const form           = document.getElementById('forecast-form');
  const tickerInput    = document.getElementById('ticker-input');
  const dropdown       = document.getElementById('ticker-dropdown');
  const tickerSelected = document.getElementById('ticker-selected');
  const clearBtn       = document.getElementById('clear-ticker');
  const submitBtn      = document.getElementById('submit-btn');
  const errorBox       = document.getElementById('error-box');
  const resultsPanel   = document.getElementById('results-panel');
  const resultsName    = document.getElementById('results-filename');

  let chart          = null;
  let selectedTicker = null;   // { symbol, name, exchange }
  let searchTimer    = null;
  let lastChartParams = null;  // stored for re-render on theme change

  // ----------------------- Ticker search -------------------- //
  tickerInput.addEventListener('input', () => {
    clearTimeout(searchTimer);
    const q = tickerInput.value.trim();
    if (q.length < 2) { dropdown.hidden = true; return; }
    searchTimer = setTimeout(() => doSearch(q), 320);
  });

  tickerInput.addEventListener('keydown', e => {
    if (e.key === 'Escape') dropdown.hidden = true;
  });

  document.addEventListener('click', e => {
    if (!e.target.closest('#ticker-input') && !e.target.closest('#ticker-dropdown')) {
      dropdown.hidden = true;
    }
  });

  clearBtn.addEventListener('click', () => {
    selectedTicker = null;
    tickerSelected.hidden = true;
    tickerInput.value = '';
    tickerInput.focus();
  });

  async function doSearch(q) {
    try {
      const res  = await fetch(`/search?q=${encodeURIComponent(q)}`);
      const list = await res.json();
      renderDropdown(list);
    } catch (_) {
      dropdown.hidden = true;
    }
  }

  function renderDropdown(list) {
    if (!list.length) { dropdown.hidden = true; return; }
    dropdown.innerHTML = list.map(r => `
      <div class="ticker-dropdown-item"
           data-symbol="${r.symbol}"
           data-name="${escHtml(r.name)}"
           data-exchange="${escHtml(r.exchange)}">
        <span class="ticker-dropdown-symbol">${r.symbol}</span>
        <span class="ticker-dropdown-name">${escHtml(r.name)}</span>
        <span class="ticker-dropdown-exch">${escHtml(r.exchange)}</span>
      </div>
    `).join('');
    dropdown.hidden = false;
    dropdown.querySelectorAll('.ticker-dropdown-item').forEach(item => {
      item.addEventListener('click', () => selectTicker({
        symbol:   item.dataset.symbol,
        name:     item.dataset.name,
        exchange: item.dataset.exchange,
      }));
    });
  }

  function selectTicker(t) {
    selectedTicker = t;
    dropdown.hidden = true;
    tickerInput.value = '';
    document.getElementById('ticker-sel-symbol').textContent = t.symbol;
    document.getElementById('ticker-sel-name').textContent   = t.name;
    document.getElementById('ticker-sel-meta').textContent   = t.exchange;
    tickerSelected.hidden = false;
    hideError();
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // ----------------------- Submit --------------------------- //
  form.addEventListener('submit', async e => {
    e.preventDefault();
    hideError();

    const ticker = selectedTicker?.symbol || tickerInput.value.trim().toUpperCase();
    if (!ticker) {
      showError('Search for a company or type a ticker symbol (e.g. AAPL) first.');
      return;
    }

    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-label').textContent = 'Fetching data';

    const loadingLabel = setTimeout(() => {
      if (submitBtn.disabled) {
        submitBtn.querySelector('.btn-label').textContent = 'Running models';
      }
    }, 2000);

    try {
      const res  = await fetch('/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ ticker }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
      renderResults(data);
    } catch (err) {
      showError(err.message);
    } finally {
      clearTimeout(loadingLabel);
      submitBtn.classList.remove('loading');
      submitBtn.disabled = false;
      submitBtn.querySelector('.btn-label').textContent = 'Run forecast';
    }
  });

  // ----------------------- Rendering ------------------------ //
  function renderResults(data) {
    resultsName.textContent = data.ticker
      ? `${data.ticker}${data.company_name ? ' · ' + data.company_name : ''}`
      : '';

    const dirEl  = document.getElementById('direction-value');
    const arrEl  = document.getElementById('direction-arrow');
    const confEl = document.getElementById('direction-confidence');
    const fill   = document.getElementById('confidence-fill');

    const dir      = data.direction || '—';
    const isUp     = dir === 'Up';
    const isDown   = dir === 'Down';
    const dirClass = isUp ? 'up' : (isDown ? 'down' : '');

    dirEl.textContent = dir;
    dirEl.classList.remove('up', 'down'); if (dirClass) dirEl.classList.add(dirClass);
    arrEl.classList.remove('up', 'down'); if (dirClass) arrEl.classList.add(dirClass);

    const conf = data.direction_confidence || 0;
    confEl.textContent = (conf * 100).toFixed(1) + '%';
    fill.classList.remove('up', 'down'); if (dirClass) fill.classList.add(dirClass);
    fill.style.width = '0%';
    requestAnimationFrame(() => { fill.style.width = (conf * 100).toFixed(1) + '%'; });

    const magEl      = document.getElementById('magnitude-value');
    const lastCloseE = document.getElementById('last-close');
    const nextCloseE = document.getElementById('next-close');
    const mag        = data.magnitude_pct ?? 0;
    const magSign    = mag >= 0 ? '+' : '';
    magEl.textContent = `${magSign}${mag.toFixed(3)}%`;
    magEl.classList.remove('up', 'down');
    magEl.classList.add(mag >= 0 ? 'up' : 'down');
    lastCloseE.textContent = fmtUsd(data.last_close);
    nextCloseE.textContent = fmtUsd(data.next_close);

    fillModelList('classifications-list', data.classifications, row => {
      const prob = (row.value * 100).toFixed(1) + '%';
      const cls  = row.value >= 0.5 ? 'up' : 'down';
      return { label: row.model, value: prob, cls };
    });
    fillModelList('regressions-list', data.regressions, row => {
      const sign = row.value >= 0 ? '+' : '';
      const cls  = row.value >= 0 ? 'up' : 'down';
      return { label: row.model, value: `${sign}${row.value.toFixed(3)}%`, cls };
    });

    drawHistoryChart(data.history, data.last_close, data.next_close);
    renderWorkflow(data);

    resultsPanel.hidden = false;
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function fillModelList(listId, rows, fmt) {
    const ul = document.getElementById(listId);
    ul.innerHTML = '';
    (rows || []).forEach(r => {
      const { label, value, cls } = fmt(r);
      const li = document.createElement('li');
      li.innerHTML = `
        <span class="model-name">${label}</span>
        <span class="model-value ${cls}">${value}</span>
      `;
      ul.appendChild(li);
    });
  }

  function getChartColors() {
    const cs = getComputedStyle(document.documentElement);
    const v  = p => cs.getPropertyValue(p).trim();
    return {
      line:    v('--chart-line'),
      fill:    v('--chart-fill'),
      tick:    v('--chart-tick'),
      grid:    v('--chart-grid'),
      up:      v('--chart-up'),
      down:    v('--chart-down'),
      tooltipBg:     v('--chart-tooltip-bg'),
      tooltipBorder: v('--chart-tooltip-border'),
      tooltipTitle:  v('--chart-tooltip-title'),
      tooltipBody:   v('--chart-tooltip-body'),
    };
  }

  function drawHistoryChart(history, lastClose, nextClose) {
    lastChartParams = { history, lastClose, nextClose };

    const canvas = document.getElementById('history-chart');
    const ctx    = canvas.getContext('2d');
    if (chart) { chart.destroy(); }

    const c = getChartColors();
    const labels  = (history || []).map(h => h.date);
    const closes  = (history || []).map(h => h.close);

    const labelsExt = [...labels, 'Next'];
    const closesExt = [...closes, null];
    const projExt   = Array(closes.length - 1).fill(null)
      .concat([lastClose, nextClose]);
    const projColor = nextClose >= lastClose ? c.up : c.down;

    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labelsExt,
        datasets: [
          {
            label: 'Close',
            data: closesExt,
            borderColor: c.line,
            backgroundColor: c.fill,
            borderWidth: 2,
            tension: 0.28,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: c.line,
            fill: true,
          },
          {
            label: 'Projected',
            data: projExt,
            borderColor: projColor,
            borderDash: [4, 4],
            borderWidth: 2,
            pointRadius: [0, 5],
            pointHoverRadius: 6,
            pointBackgroundColor: projColor,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            display: true,
            labels: {
              color: c.tick,
              font: { family: 'JetBrains Mono', size: 10 },
              usePointStyle: true,
              padding: 16,
            },
          },
          tooltip: {
            backgroundColor: c.tooltipBg,
            borderColor: c.tooltipBorder,
            borderWidth: 1,
            titleColor: c.tooltipTitle,
            bodyColor:  c.tooltipBody,
            titleFont: { family: 'JetBrains Mono', size: 11 },
            bodyFont:  { family: 'JetBrains Mono', size: 12 },
            padding: 12,
            boxPadding: 4,
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${fmtUsd(ctx.parsed.y)}`,
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: c.tick,
              font: { family: 'JetBrains Mono', size: 10 },
              maxRotation: 0,
              autoSkipPadding: 16,
            },
            grid: { color: c.grid },
          },
          y: {
            ticks: {
              color: c.tick,
              font: { family: 'JetBrains Mono', size: 10 },
              callback: v => fmtUsd(v),
            },
            grid: { color: c.grid },
          },
        },
      },
    });
  }

  // ----------------------- Workflow ------------------------- //
  function renderWorkflow(data) {
    const el = document.getElementById('workflow-section');
    if (!el) return;

    const s     = data.data_summary      || {};
    const feats = data.features_snapshot || [];
    const ens   = data.ensemble_detail   || {};
    const cls   = data.classifications   || [];
    const reg   = data.regressions       || [];

    // Step 01 — Data Ingestion
    const step1 = `
      <div class="wf-step">
        <div class="wf-step-head">
          <span class="wf-step-num">01</span>
          <div>
            <div class="wf-step-title">Data Ingestion</div>
            <div class="wf-step-sub">Live OHLCV pulled from ${data.data_source || 'Yahoo Finance'} · date-sorted and validated</div>
          </div>
        </div>
        <div class="wf-step-body">
          <div class="wf-stat-row">
            <div class="wf-stat"><span class="wf-stat-k">Ticker</span><span class="wf-stat-v">${data.ticker || '—'}</span></div>
            <div class="wf-stat"><span class="wf-stat-k">Company</span><span class="wf-stat-v">${data.company_name || '—'}</span></div>
            <div class="wf-stat"><span class="wf-stat-k">Rows</span><span class="wf-stat-v">${s.rows}</span></div>
            <div class="wf-stat"><span class="wf-stat-k">From</span><span class="wf-stat-v">${s.date_from}</span></div>
            <div class="wf-stat"><span class="wf-stat-k">To</span><span class="wf-stat-v">${s.date_to}</span></div>
          </div>
          <div class="wf-ohlcv">
            <span class="wf-ohlcv-label">Last session</span>
            <span class="wf-ohlcv-item"><em>O</em>${fmtUsd(s.last_open)}</span>
            <span class="wf-ohlcv-item"><em>H</em>${fmtUsd(s.last_high)}</span>
            <span class="wf-ohlcv-item"><em>L</em>${fmtUsd(s.last_low)}</span>
            <span class="wf-ohlcv-item"><em>C</em>${fmtUsd(s.last_close)}</span>
            <span class="wf-ohlcv-item"><em>V</em>${fmtVol(s.last_volume)}</span>
          </div>
        </div>
      </div>`;

    // Step 02 — Feature Engineering
    const featRows = feats.map(f => {
      const { text, sc } = featSignal(f.key, f.value);
      const valStr = f.value != null ? f.value.toFixed(4) : '—';
      return `<tr>
        <td class="wf-td-label">${f.label}</td>
        <td class="wf-td-value mono">${valStr}</td>
        <td class="wf-td-signal ${sc}">${text}</td>
        <td class="wf-td-desc">${f.desc}</td>
      </tr>`;
    }).join('');

    const step2 = `
      <div class="wf-step">
        <div class="wf-step-head">
          <span class="wf-step-num">02</span>
          <div>
            <div class="wf-step-title">Feature Engineering</div>
            <div class="wf-step-sub">22 technical indicators derived from OHLCV — values for the last row passed to each model</div>
          </div>
        </div>
        <div class="wf-step-body">
          <table class="wf-table">
            <thead><tr><th>Indicator</th><th>Value</th><th>Signal</th><th>Description</th></tr></thead>
            <tbody>${featRows}</tbody>
          </table>
        </div>
      </div>`;

    // Step 03 — Model Scoring
    const clsRows = cls.map(m => {
      const pct  = (m.value * 100).toFixed(2) + '%';
      const vote = m.value >= 0.5
        ? '<span class="up">↑ Up</span>'
        : '<span class="down">↓ Down</span>';
      return `<tr>
        <td class="wf-td-label">${m.model}</td>
        <td class="wf-td-value">Classification</td>
        <td class="wf-td-value mono">${pct} P(Up)</td>
        <td class="wf-td-signal">${vote}</td>
      </tr>`;
    }).join('');

    const regRows = reg.map(m => {
      const sign = m.value >= 0 ? '+' : '';
      const rc   = m.value >= 0 ? 'up' : 'down';
      return `<tr>
        <td class="wf-td-label">${m.model}</td>
        <td class="wf-td-value">Regression</td>
        <td class="wf-td-value mono ${rc}">${sign}${m.value.toFixed(4)}%</td>
        <td class="wf-td-signal">—</td>
      </tr>`;
    }).join('');

    const step3 = `
      <div class="wf-step">
        <div class="wf-step-head">
          <span class="wf-step-num">03</span>
          <div>
            <div class="wf-step-title">Model Scoring</div>
            <div class="wf-step-sub">${ens.total_classifiers || 0} classifiers predict P(Up) · ${ens.total_regressors || 0} regressors predict next-day % change</div>
          </div>
        </div>
        <div class="wf-step-body">
          <table class="wf-table">
            <thead><tr><th>Model</th><th>Task</th><th>Raw output</th><th>Direction vote</th></tr></thead>
            <tbody>${clsRows}${regRows}</tbody>
          </table>
        </div>
      </div>`;

    // Step 04 — Ensemble Aggregation
    const total    = ens.total_classifiers || 1;
    const upPct    = ((ens.votes_up / total) * 100).toFixed(0);
    const magParts = reg.map(r => `${r.value >= 0 ? '+' : ''}${r.value.toFixed(4)}%`).join(', ');

    const step4 = `
      <div class="wf-step">
        <div class="wf-step-head">
          <span class="wf-step-num">04</span>
          <div>
            <div class="wf-step-title">Ensemble Aggregation</div>
            <div class="wf-step-sub">Votes tallied · outputs averaged · confidence derived</div>
          </div>
        </div>
        <div class="wf-step-body">
          <div class="wf-ensemble-grid">
            <div class="wf-ens-block">
              <div class="wf-ens-title">Direction vote</div>
              <div class="wf-vote-bar-wrap">
                <div class="wf-vote-bar">
                  <div class="wf-vote-fill up" style="width:${upPct}%"></div>
                </div>
                <div class="wf-vote-labels">
                  <span class="up">${ens.votes_up} model${ens.votes_up !== 1 ? 's' : ''} voted Up</span>
                  <span class="down">${ens.votes_down} voted Down</span>
                </div>
              </div>
              <div class="wf-formula">
                Avg P(Up) across classifiers:<br>
                <strong>${ens.avg_prob}</strong><br><br>
                Confidence formula:<br>
                <strong>${ens.confidence_formula}</strong>
              </div>
            </div>
            <div class="wf-ens-block">
              <div class="wf-ens-title">Magnitude estimate</div>
              <div class="wf-formula">
                Regressor outputs:<br>
                <strong>${magParts}</strong><br><br>
                Mean of ${ens.total_regressors} regressors:<br>
                <strong>${ens.avg_mag >= 0 ? '+' : ''}${ens.avg_mag}%</strong>
              </div>
            </div>
          </div>
        </div>
      </div>`;

    // Step 05 — Final Projection
    const dirCls = data.direction === 'Up' ? 'up' : 'down';
    const step5  = `
      <div class="wf-step wf-step--last">
        <div class="wf-step-head">
          <span class="wf-step-num">05</span>
          <div>
            <div class="wf-step-title">Final Projection</div>
            <div class="wf-step-sub">Magnitude applied to last close to produce a price target</div>
          </div>
        </div>
        <div class="wf-step-body">
          <div class="wf-projection">${ens.projection_formula}</div>
          <div class="wf-final-row">
            <div class="wf-final-item">
              <span class="wf-final-k">Direction</span>
              <span class="wf-final-v ${dirCls}">${data.direction}</span>
            </div>
            <div class="wf-final-item">
              <span class="wf-final-k">Confidence</span>
              <span class="wf-final-v">${((data.direction_confidence || 0) * 100).toFixed(1)}%</span>
            </div>
            <div class="wf-final-item">
              <span class="wf-final-k">Expected change</span>
              <span class="wf-final-v ${dirCls}">${(data.magnitude_pct >= 0 ? '+' : '') + data.magnitude_pct.toFixed(3)}%</span>
            </div>
            <div class="wf-final-item">
              <span class="wf-final-k">Last close</span>
              <span class="wf-final-v">${fmtUsd(data.last_close)}</span>
            </div>
            <div class="wf-final-item">
              <span class="wf-final-k">Projected close</span>
              <span class="wf-final-v ${dirCls}">${fmtUsd(data.next_close)}</span>
            </div>
          </div>
        </div>
      </div>`;

    el.innerHTML = `
      <div class="wf-header">
        <h3>Analysis Walkthrough</h3>
        <span class="wf-header-sub">Step-by-step breakdown of how this forecast was computed</span>
      </div>
      ${step1}${step2}${step3}${step4}${step5}
    `;
  }

  function featSignal(key, value) {
    if (value == null) return { text: '—', sc: '' };
    if (key === 'rsi_14') {
      if (value > 70) return { text: 'Overbought', sc: 'down' };
      if (value < 30) return { text: 'Oversold',   sc: 'up'   };
      return { text: 'Neutral', sc: '' };
    }
    if (key === 'vix_level') {
      if (value > 30) return { text: 'High Fear',  sc: 'down' };
      if (value < 15) return { text: 'Low Fear',   sc: 'up'   };
      return { text: 'Moderate', sc: '' };
    }
    if (key === 'relative_volume') {
      if (value > 1.5) return { text: 'Elevated', sc: '' };
      if (value < 0.7) return { text: 'Low',      sc: '' };
      return { text: 'Normal', sc: '' };
    }
    if (key === 'close_location') {
      if (value > 0.7) return { text: 'Near high', sc: 'up'   };
      if (value < 0.3) return { text: 'Near low',  sc: 'down' };
      return { text: 'Mid-range', sc: '' };
    }
    if (key === 'daily_range_pct') {
      return value > 0.03 ? { text: 'Wide', sc: '' } : { text: 'Narrow', sc: '' };
    }
    if (key === 'volatility_10') {
      return value > 0.02 ? { text: 'High', sc: '' } : { text: 'Low', sc: '' };
    }
    if (['sma_10', 'sma_20'].includes(key)) {
      return { text: 'Trend Ref', sc: '' };
    }
    if (['day_of_week', 'month'].includes(key)) {
      return { text: 'Calendar', sc: '' };
    }
    if ([
      'momentum_10', 'daily_return', 'spy_return', 'sector_return',
      'relative_to_spy', 'relative_to_sector', 'tnx_change',
      'dxy_change', 'oil_return', 'overnight_gap', 'intraday_return',
    ].includes(key)) {
      return value > 0 ? { text: 'Positive', sc: 'up' } : { text: 'Negative', sc: 'down' };
    }
    return { text: '—', sc: '' };
  }

  function fmtVol(v) {
    if (v == null) return '—';
    if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B';
    if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    return String(v);
  }

  function fmtUsd(v) {
    if (v == null || Number.isNaN(v)) return '—';
    return '$' + Number(v).toLocaleString('en-US', {
      minimumFractionDigits: 2, maximumFractionDigits: 2,
    });
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = false;
  }
  function hideError() { errorBox.hidden = true; }

  // ----------------------- Theme toggle --------------------- //
  const themeToggle = document.getElementById('theme-toggle');

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('nexttick-theme', theme);
    themeToggle.textContent = theme === 'light' ? '☾ Dark' : '☀ Light';
    // Re-render chart with new palette if a result is already on screen
    if (lastChartParams && !resultsPanel.hidden) {
      drawHistoryChart(
        lastChartParams.history,
        lastChartParams.lastClose,
        lastChartParams.nextClose,
      );
    }
  }

  themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'light' ? 'dark' : 'light');
  });

  // Sync button label with whatever theme was restored from localStorage
  applyTheme(document.documentElement.getAttribute('data-theme') || 'dark');
})();
