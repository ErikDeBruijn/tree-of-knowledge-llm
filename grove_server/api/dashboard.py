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
  #settings input, #settings select { background: #151d2b; border: 1px solid #1e2d42;
    color: #e2e8f0; padding: 4px 8px; border-radius: 4px; width: 80px; }
</style>
</head>
<body>
<h1><a href="/" style="color:#4fd1c5;text-decoration:none">Grove</a> Playground</h1>
<div id="settings">
  <label>Max tokens <input type="number" id="max-tokens" value="200"></label>
  <label>Temperature <input type="number" id="temperature" value="0.7" step="0.1" min="0" max="2"></label>
  <label>Stream <input type="checkbox" id="stream" checked></label>
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
  const useStream = document.getElementById('stream').checked;

  const body = {
    model: 'qwen3-8b',
    messages: messages.map(m => ({role: m.role, content: m.content})),
    max_tokens: maxTokens,
    temperature: temperature,
    stream: useStream,
  };

  const t0 = performance.now();

  if (useStream) {
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
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    const tps = tokens > 0 ? (tokens / parseFloat(elapsed)).toFixed(1) : '?';
    const metaDiv = document.createElement('div');
    metaDiv.className = 'meta';
    metaDiv.textContent = tokens + ' tokens, ' + elapsed + 's, ' + tps + ' tok/s';
    div.appendChild(metaDiv);
    messages.push({role: 'assistant', content: fullText});
  } else {
    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
      });
      const data = await res.json();
      const content = data.choices?.[0]?.message?.content || 'No response';
      const usage = data.usage || {};
      const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
      const tps = usage.timing?.tokens_per_second?.toFixed(1) || '?';
      addMsg('assistant', content, usage.completion_tokens + ' tokens, ' + elapsed + 's, ' + tps + ' tok/s');
      messages.push({role: 'assistant', content: content});
    } catch(e) { addMsg('assistant', 'Error: ' + e.message); }
  }
  sendBtn.disabled = false;
  promptEl.focus();
}
</script>
</body>
</html>"""


@router.get("/playground", response_class=HTMLResponse)
async def playground():
    """Serve the chat playground."""
    return PLAYGROUND_HTML
