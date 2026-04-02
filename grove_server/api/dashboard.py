"""Dashboard: single-page live monitoring UI for Grove Server."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

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
<a href="/playground">Playground</a>
<a href="/dashboard">Dashboard</a>
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
    return DASHBOARD_HTML


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
<div id="settings">
  <label>Max tokens <input type="number" id="max-tokens" value="100"></label>
  <label>Temperature <input type="number" id="temperature" value="0.7" step="0.1" min="0" max="2"></label>
</div>
<textarea id="prompt" placeholder="Enter prompt text for completion..."></textarea>
<div id="controls">
  <button id="send" onclick="generate()">Complete</button>
  <span class="meta" id="status"></span>
</div>
<div id="output"></div>
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

function gateToColor(gates, experts) {
  if (!gates || Object.keys(gates).length === 0) return 'transparent';
  // Average gate across all layers for this token
  const values = Object.values(gates);
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  if (avg < 0.05) return 'transparent';
  // Use first expert's color (TODO: blend for multi-expert)
  const c = EXPERT_COLORS[0] || [79, 209, 197];
  const alpha = Math.min(0.6, avg * 0.8);
  return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + alpha.toFixed(2) + ')';
}

function showTooltip(e, tokenData) {
  const gates = tokenData.layer_gates || {};
  const layers = Object.keys(gates).map(Number).sort((a, b) => a - b);

  let html = '<h3>"' + tokenData.token.replace(/</g, '&lt;') + '"</h3>';

  if (layers.length === 0) {
    html += '<div class="expert-row">No expert active (base model only)</div>';
  } else {
    // Expert summary sorted by avg gate
    const avg = layers.reduce((s, l) => s + gates[l], 0) / layers.length;
    const name = expertNames[0] || 'expert';
    html += '<div class="expert-row"><span class="expert-name">' + name +
            '</span><span class="gate-val">' + avg.toFixed(3) + ' avg</span></div>';
  }

  // Layer heatmap: all 36 layers
  html += '<div class="layer-heatmap">';
  for (let i = 0; i < 36; i++) {
    const v = gates[i] || 0;
    let bg;
    if (i < 12) {
      // Identity layers (0-11): grey scale
      bg = v > 0 ? 'rgba(160,174,192,' + (v * 0.8).toFixed(2) + ')' : '#1a2332';
    } else {
      // Expert layers (12-35): teal scale
      const c = EXPERT_COLORS[0] || [79, 209, 197];
      bg = v > 0.01 ? 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + Math.max(0.1, v * 0.9).toFixed(2) + ')' : '#1a2332';
    }
    html += '<div class="layer-cell" data-idx="' + i + '" style="background:' + bg + '" title="L' + i + ': ' + v.toFixed(3) + '"></div>';
  }
  html += '</div>';
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

  try {
    const res = await fetch('/v1/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt: text, max_tokens: maxTokens, temperature: temperature }),
    });
    const data = await res.json();
    expertNames = data.experts || [];

    // Render tokens with color-coded backgrounds
    (data.tokens || []).forEach((td, idx) => {
      const span = document.createElement('span');
      span.className = 'tok';
      span.textContent = td.token;
      span.style.background = gateToColor(td.layer_gates, data.experts);
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

    const t = data.timing || {};
    statusEl.textContent = (data.tokens || []).length + ' tokens, ' +
      (t.generation_ms / 1000).toFixed(2) + 's, ' + t.tokens_per_second + ' tok/s';
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
  }
  sendBtn.disabled = false;
}

promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); generate(); }
});
</script>
</body>
</html>"""


@router.get("/playground", response_class=HTMLResponse)
async def playground():
    """Serve the chat playground."""
    return PLAYGROUND_HTML
