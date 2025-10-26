let chart = null;

function updateApiLink(container) {
  const link = document.getElementById('apiLink');
  link.href = `/${container}/forecast`;
  link.textContent = `/${container}/forecast`;
}

async function loadData(container) {
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'Lade Daten…';
  try {
    const res = await fetch(`/${container}/forecast`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    statusEl.textContent = '';
    return data;
  } catch (e) {
    statusEl.innerHTML = `<span class="error">Fehler beim Laden: ${e.message}</span>`;
    throw e;
  }
}


// Metriken anhand der aktuell gewählten Version berechnen
async function computeAndRenderMetrics(container) {
  const deviationEl = document.getElementById('deviationValue');
  const trendEl = document.getElementById('trendValue');
  try {
    const res = await fetch(`/${container}/forecast`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const arr = await res.json();
    // In EUR/kWh; wir brauchen Markt und Forecast auf gemeinsamen Startzeiten
    const market = arr.filter(d => d.price_origin === 'market');
    const forecast = arr.filter(d => d.price_origin === 'forecast');

    const marketMap = new Map(market.map(d => [d.start, d.price]));
    const paired = forecast.filter(f => marketMap.has(f.start))
      .map(f => ({ t: f.start, p_m: marketMap.get(f.start), p_f: f.price }));

    if (!paired.length) {
      deviationEl.textContent = '–';
      trendEl.textContent = '–';
      return;
    }

    // Durchschnittliche Abweichung (ct/kWh)
    const absCt = paired.map(r => Math.abs(r.p_f - r.p_m) * 100);
    const avgCt = absCt.reduce((a, b) => a + b, 0) / absCt.length;
    deviationEl.innerHTML = `${avgCt.toFixed(1)}<span class="metric-unit">ct/kWh</span>`;

    // Trend-Trefferquote
    paired.sort((a, b) => a.t.localeCompare(b.t));
    let hits = 0, total = 0;
    const eps = 1e-6;
    for (let i = 1; i < paired.length; i++) {
      const dp = paired[i].p_m - paired[i - 1].p_m;
      const dfc = paired[i].p_f - paired[i - 1].p_f;
      const sp = Math.abs(dp) <= eps ? 0 : Math.sign(dp);
      const sf = Math.abs(dfc) <= eps ? 0 : Math.sign(dfc);
      if (sp === sf) hits++;
      total++;
    }
    const rate = total > 0 ? Math.round((hits / total) * 100) : 0;
    trendEl.innerHTML = `${rate}<span class="metric-unit">%</span>`;
  } catch (e) {
    deviationEl.textContent = '–';
    trendEl.textContent = '–';
  }
}

// DailyTask-UI entfernt (Logs-Box ersetzt) – Funktion nicht mehr benötigt

function toDataset(data, origin) {
  return data
    .filter(d => d.price_origin === origin)
    .map(d => ({ x: d.start, y: d.price })); // Preis in EUR/kWh
}

function createChart(ctx, marketPoints, forecastPoints) {
  const gridColor = 'rgba(148,163,184,0.2)';
  const tickColor = '#cbd5e1';
  // Plugin: Vertikale "Jetzt"-Linie im Chart
  const nowPlugin = {
    id: 'nowLine',
    afterDraw(chart) {
      const xScale = chart.scales?.x;
      const yScale = chart.scales?.y;
      if (!xScale || !yScale) return;
      const now = new Date();
      const x = xScale.getPixelForValue(now);
      if (Number.isNaN(x) || x < xScale.left || x > xScale.right) return;
      const { ctx } = chart;
      ctx.save();
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(x, yScale.top);
      ctx.lineTo(x, yScale.bottom);
      ctx.stroke();
      // Label
      ctx.fillStyle = '#e5e7eb';
      ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      // leicht versetzt rechts oberhalb der Linie zeichnen
      ctx.fillText('Jetzt', x + 8, yScale.top + 6);
      ctx.restore();
    }
  };
  const plugins = [nowPlugin];

  return new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Marktpreis',
          data: marketPoints,
          borderColor: '#22d3ee',
          backgroundColor: 'rgba(34,211,238,0.15)',
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.25,
          fill: true,
        },
        {
          label: 'Vorhersage',
          data: forecastPoints,
          borderColor: '#a78bfa',
          backgroundColor: 'rgba(167,139,250,0.10)',
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.25,
          borderDash: [6, 6],
          fill: true,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      layout: { padding: { top: 12, bottom: 12 } },
      plugins: {
        legend: { labels: { color: '#e5e7eb' } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const val = ctx.parsed.y;
              // Anzeige in ct/kWh
              return `${ctx.dataset.label}: ${(val * 100).toFixed(2)} ct/kWh`;
            }
          }
        }
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'day',
            stepSize: 1,
            tooltipFormat: 'dd.MM.yyyy HH:mm',
            displayFormats: { day: 'dd.MM' }
          },
          grid: { color: gridColor },
          ticks: { color: tickColor, autoSkip: true, maxRotation: 0 }
        },
        y: {
          grace: '10%',
          grid: { color: gridColor },
          ticks: {
            color: tickColor,
            callback: (v) => `${(v * 100).toFixed(0)} ct`
          },
          title: { display: true, text: 'Preis [ct/kWh]', color: tickColor }
        }
      }
    },
    plugins
  });
}

(async function init() {
  const ctx = document.getElementById('priceChart').getContext('2d');
  const select = document.getElementById('containerSelect');
  const saved = localStorage.getItem('c4e_container') || 'initial';
  select.value = saved;
  updateApiLink(saved);

  async function loadAndRender(container) {
    const data = await loadData(container);
    data.sort((a, b) => a.start.localeCompare(b.start));
    const marketPoints = toDataset(data, 'market');
    const forecastPoints = toDataset(data, 'forecast');
    if (!chart) {
      chart = createChart(ctx, marketPoints, forecastPoints);
    } else {
      chart.data.datasets[0].data = marketPoints;
      chart.data.datasets[1].data = forecastPoints;
    }
    const all = [...marketPoints, ...forecastPoints];
    if (all.length) {
      const minX = all[0].x;
      const maxX = all[all.length - 1].x;
      chart.options.scales.x.min = minX;
      chart.options.scales.x.max = maxX;
    }
    chart.update();
  }

  try {
    await loadAndRender(saved);
    await computeAndRenderMetrics(saved);
    setInterval(() => chart && chart.update('none'), 60000);
  } catch (e) {
    console.error(e);
  }

  select.addEventListener('change', async () => {
    const container = select.value;
    localStorage.setItem('c4e_container', container);
    updateApiLink(container);
    try {
      await loadAndRender(container);
      // Metriken passend zur gewählten Version berechnen
      await computeAndRenderMetrics(container);
    } catch (e) { console.error(e); }
  });

})();
