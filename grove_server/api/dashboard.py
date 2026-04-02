"""Dashboard: single-page live monitoring UI for Grove Server."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


def _nav(active: str = "") -> str:
    """Generate nav bar HTML. Active page gets teal color."""
    links = [
        ("Completion", "/playground"),
        ("Chat", "/playground/chat"),
        ("Dashboard", "/dashboard"),
        ("API Docs", "/api/docs"),
    ]
    parts = []
    for label, href in links:
        color = "#4fd1c5" if label == active else "#6b7fa3"
        parts.append(f'<a href="{href}" style="color:{color};margin-right:16px;">{label}</a>')
    return f'<nav style="margin-bottom:12px;font-size:0.85em;font-family:-apple-system,sans-serif;">{"".join(parts)}</nav>'


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grove Server Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0e17; color: #c8d6e5; padding: 20px; }
  h1 { color: #4fd1c5; margin-bottom: 16px; font-size: 1.6em; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1000px; }
  .card { background: #151d2b; border: 1px solid #1e2d42; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 0.9em; color: #6b7fa3; text-transform: uppercase;
             letter-spacing: 0.05em; margin-bottom: 10px; }
  .full-width { grid-column: 1 / -1; }

  /* Mode indicator */
  #current-mode { font-size: 2em; font-weight: bold; }
  .mode-training { color: #48bb78; }
  .mode-inference { color: #4299e1; }
  .mode-idle { color: #718096; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
  .pulse { animation: pulse 1.5s infinite; }

  #contributing { color: #a0aec0; font-size: 0.95em; margin-top: 6px; }

  /* Stats */
  .stat { display: inline-block; margin-right: 24px; margin-bottom: 8px; }
  .stat .value { font-size: 1.5em; color: #e2e8f0; font-weight: bold; }
  .stat .label { font-size: 0.75em; color: #6b7fa3; }

  /* Chart */
  #chart-container { position: relative; width: 100%; height: 150px; }
  #loss-chart { width: 100%; height: 100%; }

  /* Gate heatmap */
  .heatmap-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 3px; }
  .gate-cell { aspect-ratio: 1; border-radius: 3px; min-height: 20px;
               background: #1a2332; transition: background 0.3s; }
</style>
</head>
<body>
<h1><a href="/" style="color:#4fd1c5;text-decoration:none">Grove</a> Server</h1>
<!-- NAV -->
<div class="grid">

  <!-- Mode -->
  <div class="card">
    <h2>Mode</h2>
    <div id="current-mode" class="mode-idle">Idle</div>
    <div id="contributing"></div>
  </div>

  <!-- Performance -->
  <div class="card">
    <h2>Performance</h2>
    <div class="stat"><div class="value" id="tok-s">0</div><div class="label">tok/s inference</div></div>
    <div class="stat"><div class="value" id="steps-s">0</div><div class="label">steps total</div></div>
  </div>

  <!-- Counters -->
  <div class="card">
    <h2>Counters</h2>
    <div class="stat"><div class="value" id="inf-count">0</div><div class="label">inference requests</div></div>
    <div class="stat"><div class="value" id="train-steps">0</div><div class="label">training steps</div></div>
    <div class="stat"><div class="value" id="switches">0</div><div class="label">mode switches</div></div>
  </div>

  <!-- Loss -->
  <div class="card">
    <h2>Loss (avg)</h2>
    <div class="stat"><div class="value" id="avg-loss">-</div><div class="label">current avg</div></div>
  </div>

  <!-- Loss chart -->
  <div class="card full-width">
    <h2>Loss Curve (last 100 steps)</h2>
    <div id="chart-container">
      <canvas id="loss-chart"></canvas>
    </div>
  </div>

  <!-- Gate heatmap -->
  <div class="card full-width">
    <h2>Gate Activation Heatmap (36 layers)</h2>
    <div class="heatmap-grid" id="gate-heatmap"></div>
  </div>

</div>

<script>
// Init gate heatmap cells
const heatmap = document.getElementById('gate-heatmap');
for (let i = 0; i < 36; i++) {
  const cell = document.createElement('div');
  cell.className = 'gate-cell';
  cell.title = 'Layer ' + i;
  heatmap.appendChild(cell);
}

// Chart setup
const canvas = document.getElementById('loss-chart');
const ctx = canvas.getContext('2d');

function drawChart(losses) {
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight;
  ctx.clearRect(0, 0, w, h);
  if (!losses || losses.length < 2) return;

  const min = Math.min(...losses) * 0.95;
  const max = Math.max(...losses) * 1.05 || 1;
  const range = max - min || 1;

  ctx.strokeStyle = '#4fd1c5';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < losses.length; i++) {
    const x = (i / (losses.length - 1)) * w;
    const y = h - ((losses[i] - min) / range) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function updateGateHeatmap(gateActivations) {
  const cells = heatmap.children;
  if (!gateActivations || gateActivations.length === 0) return;
  for (let i = 0; i < 36 && i < gateActivations.length; i++) {
    const v = Math.min(1, Math.max(0, gateActivations[i]));
    const r = Math.round(79 + v * 176);
    const g = Math.round(209 + v * (-80));
    const b = Math.round(197 + v * (-140));
    cells[i].style.background = 'rgb(' + r + ',' + g + ',' + b + ')';
  }
}

// Poll metrics
async function poll() {
  try {
    const resp = await fetch('/v1/metrics');
    const d = await resp.json();

    // Mode
    const modeEl = document.getElementById('current-mode');
    const mode = d.current_mode || 'idle';
    modeEl.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
    modeEl.className = 'mode-' + mode;
    if (mode === 'training') modeEl.classList.add('pulse');

    // Contributing
    const contrib = document.getElementById('contributing');
    if (d.adapter_name) {
      contrib.textContent = 'Contributing to: ' + d.adapter_name +
        ' (step ' + d.training_steps + ', phase ' + (d.phase || '?') + ')';
    } else {
      contrib.textContent = '';
    }

    // Stats
    document.getElementById('tok-s').textContent =
      d.tokens_per_second ? d.tokens_per_second.toFixed(1) : '0';
    document.getElementById('steps-s').textContent = d.training_steps || 0;
    document.getElementById('inf-count').textContent = d.inference_requests || 0;
    document.getElementById('train-steps').textContent = d.training_steps || 0;
    document.getElementById('switches').textContent = d.switches || 0;
    document.getElementById('avg-loss').textContent =
      d.avg_loss !== null && d.avg_loss !== undefined ? d.avg_loss.toFixed(4) : '-';

    // Chart
    drawChart(d.training_losses || []);

    // Gate heatmap
    if (d.gate_activations) updateGateHeatmap(d.gate_activations);

  } catch (e) { /* retry next tick */ }
}

setInterval(poll, 1000);
poll();
</script>
</body>
</html>"""


INDEX_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Grove Server</title>
<style>
body { font-family: -apple-system, sans-serif; background: #0a0e17; color: #c8d6e5;
       display: flex; justify-content: center; align-items: center; height: 100vh; }
.links { text-align: center; }
h1 { color: #4fd1c5; margin-bottom: 24px; }
a { display: inline-block; margin: 8px 16px; padding: 16px 32px; background: #151d2b;
    border: 1px solid #1e2d42; border-radius: 8px; color: #4fd1c5; text-decoration: none;
    font-size: 1.2em; transition: background 0.2s; }
a:hover { background: #1e2d42; }
</style></head><body>
<div class="links">
<h1>Grove Server</h1>
<a href="/playground">Completion</a>
<a href="/playground/chat">Chat</a>
<a href="/dashboard">Dashboard</a>
<a href="/api/docs">API Docs</a>
<a href="/v1/health">Health</a>
<a href="/v1/metrics">Metrics</a>
</div></body></html>"""


@router.get("/", response_class=HTMLResponse)
async def index():
    """Landing page with links."""
    return INDEX_HTML


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the live dashboard."""
    return DASHBOARD_HTML.replace("<!-- NAV -->", _nav("Dashboard"))


PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grove Playground</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace;
         background: #0a0e17; color: #c8d6e5; padding: 20px; }
  h1 { color: #4fd1c5; margin-bottom: 12px; font-size: 1.4em; font-family: -apple-system, sans-serif; }
  #settings { display: flex; gap: 16px; margin-bottom: 12px; font-size: 0.85em;
              font-family: -apple-system, sans-serif; }
  #settings label { color: #6b7fa3; }
  #settings input { background: #151d2b; border: 1px solid #1e2d42;
    color: #e2e8f0; padding: 4px 8px; border-radius: 4px; width: 80px; }

  /* Prompt area */
  #prompt { width: 100%; min-height: 80px; padding: 12px; background: #151d2b;
            border: 1px solid #1e2d42; border-radius: 8px; color: #e2e8f0;
            font-family: inherit; font-size: 0.95em; resize: vertical; outline: none; }
  #prompt:focus { border-color: #4fd1c5; }
  #controls { display: flex; gap: 8px; margin: 8px 0; align-items: center; }
  button { padding: 8px 20px; background: #4fd1c5; color: #0a0e17; border: none;
           border-radius: 6px; font-weight: bold; cursor: pointer; font-family: -apple-system, sans-serif; }
  button:hover { background: #38b2ac; }
  button:disabled { background: #2d3748; color: #718096; cursor: wait; }
  .meta { font-size: 0.8em; color: #718096; font-family: -apple-system, sans-serif; }

  /* Token output */
  #output { margin-top: 12px; padding: 16px; background: #151d2b; border: 1px solid #1e2d42;
            border-radius: 8px; min-height: 100px; line-height: 1.8; font-size: 1em; }
  .tok { display: inline; cursor: default; padding: 1px 0; border-radius: 2px;
         transition: background 0.15s; position: relative; }
  .tok:hover { outline: 1px solid #4fd1c550; }
  .tok-thinking { color: #8b8fa3; font-style: italic; opacity: 0.7; }

  /* Tooltip with layer heatmap */
  .tooltip { display: none; position: fixed; background: #1a2332; border: 1px solid #2d4a6a;
             border-radius: 8px; padding: 12px; z-index: 100; pointer-events: none;
             font-family: -apple-system, sans-serif; font-size: 0.8em;
             box-shadow: 0 4px 20px rgba(0,0,0,0.5); max-width: 320px; }
  .tooltip.visible { display: block; }
  .tooltip h3 { color: #4fd1c5; font-size: 0.9em; margin-bottom: 8px; }
  .tooltip .expert-row { margin-bottom: 4px; color: #a0aec0; }
  .tooltip .expert-name { color: #e2e8f0; font-weight: bold; }
  .tooltip .gate-val { color: #4fd1c5; float: right; }

  /* Layer heatmap grid */
  .layer-heatmap { display: grid; grid-template-columns: repeat(12, 1fr); gap: 2px;
                   margin-top: 8px; }
  .layer-cell { width: 100%; aspect-ratio: 1; border-radius: 2px; min-width: 14px;
                min-height: 14px; position: relative; }
  .layer-cell::after { content: attr(data-idx); position: absolute; inset: 0;
                       display: flex; align-items: center; justify-content: center;
                       font-size: 7px; color: rgba(255,255,255,0.4); }
  .heatmap-label { font-size: 0.7em; color: #6b7fa3; margin-top: 4px; }
  .heatmap-labels { display: flex; justify-content: space-between; }
</style>
</head>
<body>
<h1><a href="/" style="color:#4fd1c5;text-decoration:none">Grove</a> Playground</h1>
<!-- NAV -->
<div id="settings">
  <label>Max tokens <input type="number" id="max-tokens" value="100"></label>
  <label>Temperature <input type="number" id="temperature" value="0.7" step="0.1" min="0" max="2"></label>
</div>
<div id="expert-checkboxes" style="margin-bottom:8px;font-size:0.85em;font-family:-apple-system,sans-serif;color:#6b7fa3;">
  Experts: <span id="expert-list">loading...</span>
</div>
<textarea id="prompt" placeholder="Enter prompt text for completion..."></textarea>
<div id="controls">
  <button id="send" onclick="generate()">Complete</button>
  <span class="meta" id="status"></span>
</div>
<div id="output"></div>
<div id="summary" style="display:none;margin-top:12px;padding:12px;background:#151d2b;border:1px solid #1e2d42;border-radius:8px;font-family:-apple-system,sans-serif;font-size:0.85em;">
  <div style="color:#6b7fa3;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;font-size:0.8em;">Response Completion Summary</div>
  <div id="summary-experts"></div>
  <div id="summary-heatmap"></div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
const promptEl = document.getElementById('prompt');
const outputEl = document.getElementById('output');
const statusEl = document.getElementById('status');
const sendBtn = document.getElementById('send');
const tooltip = document.getElementById('tooltip');

// Expert color palette (distinct, colorblind-friendly-ish)
const EXPERT_COLORS = [
  [79, 209, 197],   // teal
  [237, 137, 54],   // orange
  [159, 122, 234],  // purple
  [236, 201, 75],   // yellow
  [72, 187, 120],   // green
  [245, 101, 101],  // red
  [99, 179, 237],   // blue
  [237, 100, 166],  // pink
];
const BASE_COLOR = [30, 45, 66]; // dark blue-grey for no expert

let expertNames = [];

// Load expert checkboxes
async function loadExpertCheckboxes() {
  try {
    const resp = await fetch('/v1/experts');
    const data = await resp.json();
    const container = document.getElementById('expert-list');
    if (!data.experts || data.experts.length === 0) {
      container.textContent = 'none loaded';
      return;
    }
    container.innerHTML = '';
    data.experts.forEach((name, i) => {
      const label = document.createElement('label');
      label.style.cssText = 'margin-right:12px;color:#e2e8f0;cursor:pointer;';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.value = name;
      cb.id = 'expert-cb-' + i;
      cb.style.cssText = 'margin-right:4px;accent-color:#4fd1c5;';
      label.appendChild(cb);
      label.appendChild(document.createTextNode(name));
      container.appendChild(label);
    });
  } catch(e) {
    document.getElementById('expert-list').textContent = 'error loading';
  }
}
loadExpertCheckboxes();

function getSelectedExperts() {
  const cbs = document.querySelectorAll('#expert-list input[type=checkbox]');
  return Array.from(cbs).filter(cb => cb.checked).map(cb => cb.value);
}

function gateToColor(tokenData, experts) {
  const eg = tokenData.expert_gates || {};
  const lg = tokenData.layer_gates || {};
  // Multi-expert: blend colors by each expert's avg gate
  const expertEntries = Object.entries(eg);
  if (expertEntries.length === 0) {
    // Fallback to layer_gates
    const values = Object.values(lg);
    if (values.length === 0) return 'transparent';
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    if (avg < 0.05) return 'transparent';
    const c = EXPERT_COLORS[0] || [79, 209, 197];
    return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + Math.min(0.6, avg * 0.8).toFixed(2) + ')';
  }
  // Blend: mix RGB weighted by each expert's average gate
  let r = 0, g = 0, b = 0, totalWeight = 0;
  expertEntries.forEach(([name, layers], i) => {
    const vals = Object.values(layers);
    if (vals.length === 0) return;
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    const c = EXPERT_COLORS[i % EXPERT_COLORS.length];
    r += c[0] * avg; g += c[1] * avg; b += c[2] * avg;
    totalWeight += avg;
  });
  if (totalWeight < 0.05) return 'transparent';
  r /= totalWeight; g /= totalWeight; b /= totalWeight;
  const alpha = Math.min(0.6, totalWeight * 0.4);
  return 'rgba(' + Math.round(r) + ',' + Math.round(g) + ',' + Math.round(b) + ',' + alpha.toFixed(2) + ')';
}

function showTooltip(e, tokenData) {
  const eg = tokenData.expert_gates || {};
  const lg = tokenData.layer_gates || {};
  const expertEntries = Object.entries(eg);

  let html = '<h3>"' + tokenData.token.replace(/</g, '&lt;') + '"</h3>';

  if (expertEntries.length === 0 && Object.keys(lg).length === 0) {
    html += '<div class="expert-row">No expert active (base model only)</div>';
  }

  // Show each expert with avg gate, sorted by activation
  const expertAvgs = expertEntries.map(([name, layers], i) => {
    const vals = Object.values(layers);
    const avg = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    return { name, layers, avg, colorIdx: i };
  }).sort((a, b) => b.avg - a.avg);

  expertAvgs.forEach(({ name, avg, colorIdx }) => {
    const c = EXPERT_COLORS[colorIdx % EXPERT_COLORS.length];
    const dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:rgb(' + c.join(',') + ');margin-right:6px;"></span>';
    html += '<div class="expert-row">' + dot + '<span class="expert-name">' + name +
            '</span><span class="gate-val">' + avg.toFixed(3) + '</span></div>';
  });

  // Layer heatmap per expert (or combined if single)
  const heatmapExperts = expertAvgs.length > 0 ? expertAvgs : [{ name: 'base', layers: lg, avg: 0, colorIdx: 0 }];
  heatmapExperts.forEach(({ name, layers, colorIdx }) => {
    if (expertAvgs.length > 1) {
      const c = EXPERT_COLORS[colorIdx % EXPERT_COLORS.length];
      html += '<div style="font-size:0.7em;color:rgb(' + c.join(',') + ');margin-top:6px;">' + name + '</div>';
    }
    html += '<div class="layer-heatmap">';
    for (let i = 0; i < 36; i++) {
      const v = layers[i] || 0;
      const c = EXPERT_COLORS[colorIdx % EXPERT_COLORS.length];
      let bg;
      if (i < 12) {
        bg = v > 0 ? 'rgba(160,174,192,' + (v * 0.8).toFixed(2) + ')' : '#1a2332';
      } else {
        bg = v > 0.01 ? 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + Math.max(0.1, v * 0.9).toFixed(2) + ')' : '#1a2332';
      }
      html += '<div class="layer-cell" data-idx="' + i + '" style="background:' + bg + '" title="L' + i + ': ' + v.toFixed(3) + '"></div>';
    }
    html += '</div>';
  });
  html += '<div class="heatmap-labels"><span class="heatmap-label">L0 identity</span><span class="heatmap-label">L12 expert →</span><span class="heatmap-label">L35</span></div>';

  tooltip.innerHTML = html;
  tooltip.classList.add('visible');

  // Position near cursor
  const x = Math.min(e.clientX + 12, window.innerWidth - 340);
  const y = Math.min(e.clientY + 12, window.innerHeight - 250);
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}

function hideTooltip() {
  tooltip.classList.remove('visible');
}

async function generate() {
  const text = promptEl.value;
  if (!text) return;
  sendBtn.disabled = true;
  statusEl.textContent = 'Generating...';
  outputEl.innerHTML = '';

  const maxTokens = parseInt(document.getElementById('max-tokens').value) || 100;
  const temperature = parseFloat(document.getElementById('temperature').value) || 0.7;

  const selectedExperts = getSelectedExperts();

  try {
    const res = await fetch('/v1/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: text, max_tokens: maxTokens, temperature: temperature,
        experts: selectedExperts,
      }),
    });
    const data = await res.json();
    expertNames = data.experts || [];

    // Render tokens with color-coded backgrounds
    const allTokenData = data.tokens || [];
    allTokenData.forEach((td, idx) => {
      const span = document.createElement('span');
      span.className = 'tok';
      span.textContent = td.token;
      // Style thinking tokens differently
      const isThinkTag = td.token.includes('<think>') || td.token.includes('</think>');
      if (isThinkTag) {
        span.style.background = 'rgba(107,127,163,0.3)';
        span.style.color = '#6b7fa3';
        span.style.fontStyle = 'italic';
      } else {
        span.style.background = gateToColor(td, data.experts);
      }
      span.addEventListener('mouseenter', e => showTooltip(e, td));
      span.addEventListener('mousemove', e => {
        const x = Math.min(e.clientX + 12, window.innerWidth - 340);
        const y = Math.min(e.clientY + 12, window.innerHeight - 250);
        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
      });
      span.addEventListener('mouseleave', hideTooltip);
      outputEl.appendChild(span);
    });

    // Response-level summary
    renderSummary(allTokenData, expertNames);

    const t = data.timing || {};
    statusEl.textContent = allTokenData.length + ' tokens, ' +
      (t.generation_ms / 1000).toFixed(2) + 's, ' + t.tokens_per_second + ' tok/s';
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
  }
  sendBtn.disabled = false;
}

function renderSummary(tokens, experts) {
  const summaryEl = document.getElementById('summary');
  const expertsEl = document.getElementById('summary-experts');
  const heatmapEl = document.getElementById('summary-heatmap');

  if (!tokens.length) { summaryEl.style.display = 'none'; return; }
  summaryEl.style.display = 'block';

  // Collect per-expert, per-layer averages across all tokens
  const expertData = {};  // name -> {layer -> [values]}
  tokens.forEach(td => {
    Object.entries(td.expert_gates || {}).forEach(([ename, layers]) => {
      if (!expertData[ename]) expertData[ename] = {};
      Object.entries(layers).forEach(([l, v]) => {
        if (!expertData[ename][l]) expertData[ename][l] = [];
        expertData[ename][l].push(v);
      });
    });
  });

  // Build summary + heatmap per expert
  let expertsHtml = '';
  let heatmapHtml = '';
  const expertNames = Object.keys(expertData);

  expertNames.sort((a, b) => {
    const avgA = Object.values(expertData[a]).reduce((s, vs) => s + vs.reduce((a,b)=>a+b,0)/vs.length, 0) / Object.keys(expertData[a]).length;
    const avgB = Object.values(expertData[b]).reduce((s, vs) => s + vs.reduce((a,b)=>a+b,0)/vs.length, 0) / Object.keys(expertData[b]).length;
    return avgB - avgA;
  });

  expertNames.forEach((ename, idx) => {
    const layers = expertData[ename];
    const layerAvgs = {};
    Object.entries(layers).forEach(([l, vs]) => { layerAvgs[l] = vs.reduce((a,b)=>a+b,0)/vs.length; });
    const allVals = Object.values(layerAvgs);
    const avg = allVals.length > 0 ? allVals.reduce((a,b)=>a+b,0)/allVals.length : 0;
    const c = EXPERT_COLORS[idx % EXPERT_COLORS.length];
    const dot = '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:rgb(' + c.join(',') + ');margin-right:6px;"></span>';

    expertsHtml += '<div style="margin-bottom:4px;">' + dot +
      '<span style="color:#e2e8f0;font-weight:bold;">' + ename + '</span> ' +
      '<span style="color:rgb(' + c.join(',') + ');">' + avg.toFixed(3) + ' avg gate</span>' +
      ' <span style="color:#718096;">(' + tokens.length + ' tokens)</span></div>';

    // Heatmap for this expert
    heatmapHtml += '<div style="font-size:0.7em;color:rgb(' + c.join(',') + ');margin-top:6px;">' + ename + '</div>';
    heatmapHtml += '<div class="layer-heatmap">';
    for (let i = 0; i < 36; i++) {
      const v = layerAvgs[i] || 0;
      let bg;
      if (i < 12) {
        bg = v > 0 ? 'rgba(160,174,192,' + (v * 0.8).toFixed(2) + ')' : '#1a2332';
      } else {
        bg = v > 0.01 ? 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + Math.max(0.1, v * 0.9).toFixed(2) + ')' : '#1a2332';
      }
      heatmapHtml += '<div class="layer-cell" data-idx="' + i + '" style="background:' + bg + '" title="L' + i + ': ' + v.toFixed(3) + '"></div>';
    }
    heatmapHtml += '</div>';
  });

  if (!expertNames.length) {
    expertsHtml = '<span style="color:#718096;">No expert active</span>';
  }
  heatmapHtml += '<div class="heatmap-labels"><span class="heatmap-label">L0 identity</span><span class="heatmap-label">L12 expert →</span><span class="heatmap-label">L35</span></div>';

  expertsEl.innerHTML = expertsHtml;
  heatmapEl.innerHTML = heatmapHtml;
}

promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); generate(); }
});

// Auto-fill from URL param and run
const urlParams = new URLSearchParams(window.location.search);
const urlPrompt = urlParams.get('prompt');
if (urlPrompt) {
  promptEl.value = urlPrompt;
  // Auto-generate after experts load
  setTimeout(() => generate(), 500);
}
</script>
</body>
</html>"""


@router.get("/playground", response_class=HTMLResponse)
async def playground():
    """Serve the completion playground."""
    return PLAYGROUND_HTML.replace("<!-- NAV -->", _nav("Completion"))


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grove Chat</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0e17; color: #c8d6e5; display: flex; flex-direction: column;
         height: 100vh; padding: 16px; }
  h1 { color: #4fd1c5; margin-bottom: 12px; font-size: 1.4em; }
  #chat { flex: 1; overflow-y: auto; padding: 12px; background: #151d2b;
          border: 1px solid #1e2d42; border-radius: 8px; margin-bottom: 12px; }
  .msg { margin-bottom: 12px; padding: 10px 14px; border-radius: 8px; max-width: 80%;
         line-height: 1.5; white-space: pre-wrap; }
  .user { background: #1a365d; margin-left: auto; text-align: right; }
  .assistant { background: #1c2e3a; color: #e2e8f0; }
  .thinking { background: #1a1a2e; color: #8b8fa3; border-left: 3px solid #6b7fa3;
              padding: 8px 12px; margin-bottom: 8px; font-size: 0.85em;
              font-style: italic; border-radius: 4px; }
  .meta { font-size: 0.75em; color: #718096; margin-top: 4px; }
  #input-area { display: flex; gap: 8px; }
  #prompt { flex: 1; padding: 10px 14px; background: #151d2b; border: 1px solid #1e2d42;
            border-radius: 8px; color: #e2e8f0; font-size: 1em; outline: none; }
  #prompt:focus { border-color: #4fd1c5; }
  button { padding: 10px 20px; background: #4fd1c5; color: #0a0e17; border: none;
           border-radius: 8px; font-weight: bold; cursor: pointer; }
  button:hover { background: #38b2ac; }
  button:disabled { background: #2d3748; color: #718096; cursor: wait; }
  .streaming { opacity: 0.7; }
  #settings { display: flex; gap: 16px; margin-bottom: 12px; font-size: 0.85em; }
  #settings label { color: #6b7fa3; }
  #settings input { background: #151d2b; border: 1px solid #1e2d42;
    color: #e2e8f0; padding: 4px 8px; border-radius: 4px; width: 80px; }
</style>
</head>
<body>
<h1><a href="/" style="color:#4fd1c5;text-decoration:none">Grove</a> Chat</h1>
<!-- NAV -->
<div id="settings">
  <label>Max tokens <input type="number" id="max-tokens" value="500"></label>
  <label>Temperature <input type="number" id="temperature" value="0.7" step="0.1" min="0" max="2"></label>
</div>
<div id="chat"></div>
<div id="input-area">
  <input type="text" id="prompt" placeholder="Ask something..." autofocus>
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const messages = [];

promptEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });

function addMsg(role, content, meta) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = content;
  if (meta) { const m = document.createElement('div'); m.className = 'meta'; m.textContent = meta; div.appendChild(m); }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function send() {
  const text = promptEl.value.trim();
  if (!text) return;
  promptEl.value = '';
  sendBtn.disabled = true;

  messages.push({role: 'user', content: text});
  addMsg('user', text);

  const maxTokens = parseInt(document.getElementById('max-tokens').value) || 200;
  const temperature = parseFloat(document.getElementById('temperature').value) || 0.7;

  const body = {
    model: 'qwen3-8b',
    messages: messages.map(m => ({role: m.role, content: m.content})),
    max_tokens: maxTokens,
    temperature: temperature,
    stream: true,
  };

  const t0 = performance.now();
  const div = document.createElement('div');
  div.className = 'msg assistant streaming';
  const textNode = document.createElement('span');
  div.appendChild(textNode);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  let fullText = '';
  let tokens = 0;
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      for (const line of chunk.split('\\n')) {
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
          try {
            const data = JSON.parse(line.slice(6));
            const c = data.choices?.[0]?.delta?.content;
            if (c) { fullText += c; tokens++; textNode.textContent = fullText; chat.scrollTop = chat.scrollHeight; }
          } catch(e) {}
        }
      }
    }
  } catch(e) { fullText = 'Error: ' + e.message; textNode.textContent = fullText; }
  div.classList.remove('streaming');
  // Parse <think>...</think> blocks and render them visually
  const thinkMatch = fullText.match(/<think>([\s\S]*?)<\/think>/);
  if (thinkMatch) {
    const thinkContent = thinkMatch[1].trim();
    const responseContent = fullText.replace(/<think>[\s\S]*?<\/think>/, '').trim();
    div.innerHTML = '';
    if (thinkContent) {
      const thinkDiv = document.createElement('div');
      thinkDiv.className = 'thinking';
      thinkDiv.textContent = thinkContent;
      div.appendChild(thinkDiv);
    }
    const responseSpan = document.createElement('span');
    responseSpan.textContent = responseContent;
    div.appendChild(responseSpan);
    fullText = responseContent; // for messages history
  }
  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  const tps = tokens > 0 ? (tokens / parseFloat(elapsed)).toFixed(1) : '?';
  const metaDiv = document.createElement('div');
  metaDiv.className = 'meta';
  metaDiv.textContent = tokens + ' tokens, ' + elapsed + 's, ' + tps + ' tok/s  ';
  const viewLink = document.createElement('a');
  viewLink.textContent = 'View in Completion';
  viewLink.href = '#';
  viewLink.style.cssText = 'color:#4fd1c5;text-decoration:underline;cursor:pointer;';
  viewLink.addEventListener('click', async (ev) => {
    ev.preventDefault();
    // Send messages WITHOUT the last assistant response so completion
    // re-generates from the same prompt point with attribution
    const msgsForTemplate = messages.slice(0, -1);
    try {
      const res = await fetch('/v1/chat/template', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ messages: msgsForTemplate.map(m => ({role: m.role, content: m.content})) }),
      });
      const data = await res.json();
      // Navigate to completion playground with this prompt
      window.location.href = '/playground?prompt=' + encodeURIComponent(data.prompt);
    } catch(e) { alert('Error: ' + e.message); }
  });
  metaDiv.appendChild(viewLink);
  div.appendChild(metaDiv);
  messages.push({role: 'assistant', content: fullText});
  sendBtn.disabled = false;
  promptEl.focus();
}
</script>
</body>
</html>"""


@router.get("/playground/chat", response_class=HTMLResponse)
async def playground_chat():
    """Serve the streaming chat playground."""
    return CHAT_HTML.replace("<!-- NAV -->", _nav("Chat"))
