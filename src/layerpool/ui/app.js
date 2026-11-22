(() => {
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.get("o")) {
    history.replaceState(null, "", location.pathname);
    fetch("/api/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code: params.get("o") }) })
      .then((r) => r.json())
      .then((got) => { if (got.token) sessionStorage.setItem("layerpool-token", got.token); location.reload(); })
      .catch(() => location.reload());
    return;
  }
  if (params.get("t")) {
    sessionStorage.setItem("layerpool-token", params.get("t"));
    history.replaceState(null, "", location.pathname);
  }
  window.addEventListener("hashchange", () => location.reload());
  const token = sessionStorage.getItem("layerpool-token") || "";
  const $ = (id) => document.getElementById(id);
  const shades = ["#3e5f52", "#8a6d3b", "#5a5e85", "#8b4e5d", "#4f7d84", "#6b6b45"];
  const state = { status: null, devices: [], history: [], events: [], preview: {}, open: {}, dirty: {} };

  async function api(path, body) {
    const res = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: { Authorization: "Bearer " + token, "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await res.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { error: "the pool sent an unreadable answer" }; }
    if (!res.ok) {
      const err = new Error(data.error || res.statusText);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  const num = (v) => v !== null && v !== undefined && !Number.isNaN(Number(v));
  const fmt = (v, d = 3) => (num(v) ? Number(v).toFixed(d) : "–");
  const gb = (v) => (num(v) ? `${Number(v).toFixed(1)} GB` : "n/a");
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const ago = (s) => (s < 60 ? `${Math.round(s)} s ago` : s < 3600 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`);
  function clock(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`;
  }
  function compact(v, digits) {
    if (!num(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
    return Number(v).toFixed(digits);
  }

  const tip = $("tip");
  document.addEventListener("mousemove", (ev) => {
    const el = ev.target.closest && ev.target.closest("[data-tip]");
    if (!el) { if (!tip.dataset.chart) tip.hidden = true; return; }
    tip.dataset.chart = "";
    showTip(el.dataset.tip, ev.clientX, ev.clientY);
  });
  function showTip(text, x, y) {
    tip.textContent = text;
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(window.innerWidth - w - 8, x + 14)}px`;
    tip.style.top = `${y + h + 20 > window.innerHeight ? y - h - 12 : y + 16}px`;
  }

  function niceTicks(lo, hi, count) {
    if (lo === hi) { const pad = Math.abs(lo) * 0.1 || 1; lo -= pad; hi += pad; }
    const raw = (hi - lo) / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => raw <= s) || 10 * mag;
    const start = Math.floor(lo / step) * step, end = Math.ceil(hi / step) * step;
    const ticks = [];
    for (let v = start; v <= end + step / 2; v += step) ticks.push(Number(v.toFixed(10)));
    return ticks;
  }

  function chart(box, spec, force) {
    const rows = state.history.filter((h) => num(h[spec.key]));
    const last = rows[rows.length - 1];
    const key = `${rows.length}:${last ? last.step : ""}:${box.clientWidth}`;
    if (!force && box.dataset.key === key) return;
    box.dataset.key = key;
    const head = `<div class="chart-head"><span class="name">${spec.title} <span class="hint" data-tip="${esc(spec.about)}">?</span></span>
      <span class="now">${last ? `${spec.title.toLowerCase()} ${compact(last[spec.key], spec.digits)} at step ${last.step}` : ""}</span></div>`;
    if (!rows.length) {
      box.innerHTML = head + `<div class="empty-note">No steps yet.<br>The chart fills in as training runs: one point per step.</div>`;
      return;
    }
    const W = Math.max(260, box.clientWidth - 24), H = 220, L = 54, R = 12, T = 10, B = 36;
    const xs = rows.map((r) => r.step), ys = rows.map((r) => Number(r[spec.key]));
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    const yt = niceTicks(Math.min(...ys), Math.max(...ys), 4);
    const y0 = yt[0], y1 = yt[yt.length - 1];
    const X = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
    const Y = (v) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B);
    const xt = niceTicks(x0, x1, Math.min(6, Math.max(1, x1 - x0))).filter((v) => v >= x0 && v <= x1 && Number.isInteger(v));
    const path = rows.map((r, i) => `${i ? "L" : "M"}${X(r.step).toFixed(1)} ${Y(Number(r[spec.key])).toFixed(1)}`).join(" ");
    const area = `${path} L${X(rows[rows.length - 1].step).toFixed(1)} ${H - B} L${X(rows[0].step).toFixed(1)} ${H - B} Z`;
    const dots = rows.length <= 80 ? rows.map((r) => `<circle class="dot" r="2.5" cx="${X(r.step).toFixed(1)}" cy="${Y(Number(r[spec.key])).toFixed(1)}"></circle>`).join("") : "";
    box.innerHTML = head + `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(spec.title)} by step">
      <g class="grid">${yt.map((v) => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"></line>`).join("")}</g>
      <g class="axis">
        ${yt.map((v) => `<text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${compact(v, spec.tickDigits)}</text>`).join("")}
        ${xt.map((v) => `<text x="${X(v)}" y="${H - B + 16}" text-anchor="middle">${v}</text>`).join("")}
        <text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">training step</text>
        <text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(spec.unit)}</text>
      </g>
      <line class="axis-line" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"></line>
      <path class="area" d="${area}"></path><path class="series" d="${path}"></path>${dots}
      <line class="cursor" y1="${T}" y2="${H - B}" visibility="hidden"></line>
      <circle class="focus" r="4.5" visibility="hidden"></circle>
      <rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"></rect>
    </svg>`;
    const svg = box.querySelector("svg"), cursor = svg.querySelector(".cursor"), focus = svg.querySelector(".focus");
    const hit = svg.querySelector(".hit");
    hit.addEventListener("mousemove", (ev) => {
      const r = svg.getBoundingClientRect();
      const x = ((ev.clientX - r.left) / r.width) * W;
      let i = 0;
      rows.forEach((row, k) => { if (Math.abs(X(row.step) - x) < Math.abs(X(rows[i].step) - x)) i = k; });
      const row = rows[i];
      cursor.setAttribute("x1", X(row.step)); cursor.setAttribute("x2", X(row.step)); cursor.setAttribute("visibility", "visible");
      focus.setAttribute("cx", X(row.step)); focus.setAttribute("cy", Y(Number(row[spec.key]))); focus.setAttribute("visibility", "visible");
      tip.dataset.chart = "1";
      showTip(spec.describe(row), ev.clientX, ev.clientY);
    });
    hit.addEventListener("mouseleave", () => {
      cursor.setAttribute("visibility", "hidden"); focus.setAttribute("visibility", "hidden");
      tip.dataset.chart = ""; tip.hidden = true;
    });
  }

  const charts = [
    { el: "chart-loss", title: "Loss", unit: "loss", key: "loss", digits: 4, tickDigits: 2,
      about: "How far the model's predictions are from the training data, measured after each step. Lower is better; it usually falls fast at first, then slowly, and moves up and down from batch to batch.",
      describe: (r) => `Step ${r.step}\nLoss ${fmt(r.loss, 4)}\nLearning rate ${num(r.lr) ? Number(r.lr).toExponential(2) : "–"}\nGradient norm ${fmt(r.grad_norm, 3)}${r.skipped ? "\nSkipped: the gradients overflowed" : ""}` },
    { el: "chart-speed", title: "Speed", unit: "tokens per second", key: "tokens_per_s", digits: 0, tickDigits: 0,
      about: "How many tokens the whole pool trains on per second, in each step. It rises when faster devices take more layers and falls when a device is slow or the network is busy.",
      describe: (r) => `Step ${r.step}\n${Math.round(r.tokens_per_s)} tokens per second\n${r.tokens} tokens in ${fmt(r.seconds, 1)} s` },
  ];

  function tiles() {
    const s = state.status, h = state.history;
    const last = h[h.length - 1], prev = h[h.length - 2];
    const done = s && num(s.steps_done) ? s.steps_done + Math.max(0, h.length - (s.history_len || h.length)) : h.length;
    const elapsed = s && num(s.trained_s) ? s.trained_s + h.slice(s.history_len || h.length).reduce((a, r) => a + (r.seconds || 0), 0) : h.reduce((a, r) => a + (r.seconds || 0), 0);
    const change = last && prev && num(last.loss) && num(prev.loss) ? last.loss - prev.loss : null;
    const tile = (label, value, sub, about, cls = "") => `<div class="tile" data-tip="${esc(about)}"><div class="label">${label}</div><div class="value">${value}</div><div class="sub ${cls}">${sub}</div></div>`;
    const total = s && s.job && s.job.expected_steps;
    const perStep = done ? elapsed / done : 0;
    const left = total && perStep && s.step < total ? (total - s.step) * perStep : null;
    const mine = Object.entries(state.metrics || {}).filter(([k, v]) => k !== "step" && k !== "time" && num(v)).slice(0, 6);
    $("tiles").innerHTML = [
      tile("Step", s && (s.run || s.last_run) ? (total ? `${s.step} / ${total}` : s.step) : "–",
        left !== null ? `about ${clock(left)} left` : done ? `${done} done` : "waiting for the first",
        "Optimizer steps finished in this run, out of the run's length when the training code tells it (the Trainer and the built-in loops do)."),
      tile("Loss", last ? fmt(last.loss, 4) : "–", change === null ? "" : `${change <= 0 ? "▼" : "▲"} ${Math.abs(change).toFixed(4)} from the step before`,
        "The training loss of the last step. Lower is better.", change === null ? "" : change <= 0 ? "down" : "up"),
      tile("Speed", last && last.tokens_per_s ? `${compact(last.tokens_per_s, 0)}` : "–", "tokens per second", "Tokens trained on per second in the last step, by all devices together."),
      tile("Step time", last ? `${fmt(last.seconds, 1)} s` : "–", h.length ? `${clock(elapsed)} training so far` : "", "How long the last step took, and the total training time of this run."),
      tile("Learning rate", last && num(last.lr) ? Number(last.lr).toExponential(1) : "–", "", "The learning rate the last step used."),
      tile("Gradient norm", last ? fmt(last.grad_norm, 3) : "–", last && last.clip < 1 ? `clipped to ${(last.clip * 100).toFixed(0)}%` : "", "The size of the last step's gradient, over every device. Very large or growing values mean training is unstable."),
      ...mine.map(([k, v]) => tile(esc(k), fmt(v, 4), state.metrics.step !== undefined ? `at step ${esc(String(state.metrics.step))}` : "", "A value your training code logged with run.log().")),
    ].join("");
  }

  function renderRun() {
    const s = state.status;
    if (!s) return;
    $("version").textContent = "v" + s.version;
    const live = state.devices.filter((d) => d.status === "idle" || d.status === "training").length;
    const paused = state.devices.filter((d) => d.status === "disabled").length;
    $("summary").textContent = `${live} device${live === 1 ? "" : "s"} online` + (paused ? `, ${paused} paused` : "") + (s.run ? ` · training, step ${s.step}` : " · no run going");
    $("run-name").textContent = s.run || (s.last_run ? `${s.last_run} (finished)` : "No run yet");
    $("run-model").textContent = s.job ? `${s.job.repo} · ${s.job.mode === "full" ? "full fine-tuning" : "LoRA, rank " + s.job.rank}` : (s.last_run ? "Finished. Run your training code again to start a new run." : "Add devices, then run your training code: it connects with layerpool.connect() or layerpool.hf.");
    $("run-actions").hidden = !s.run;
    if (s.job && document.activeElement !== $("rebalance")) $("rebalance").checked = s.job.rebalance !== false;
    tiles();
    const pipe = $("pipe");
    const plan = s.plan || [];
    pipe.className = "pipe" + (plan.length ? "" : " empty");
    const last = state.history[state.history.length - 1];
    const timing = last && last.timing ? last.timing : {};
    pipe.innerHTML = plan.map((p, i) => {
      const n = p.range[1] - p.range[0];
      const t = timing[p.name];
      const about = `${p.name}: layers ${p.range[0]}–${p.range[1] - 1} (${n} of ${s.job ? s.job.layers : "?"})` + (t ? `\nLast step: compute ${fmt(t.compute, 2)} s · waiting ${fmt(t.wait, 2)} s · sending ${fmt(t.send, 2)} s` : "");
      return `<div class="seg" style="flex:${n};background:${shades[i % shades.length]}" data-tip="${esc(about)}">${esc(p.name)} · ${p.range[0]}–${p.range[1] - 1}</div>`;
    }).join("");
    charts.forEach((c) => chart($(c.el), c));
  }

  const statusText = {
    training: ["training", "Holds layers of the current run and trains them."],
    idle: ["idle", "Connected and ready; it has no layers in the current split (no run, or the others are faster without it)."],
    disabled: ["paused", "Paused: it takes no layers until you use it again."],
    banned: ["left out", "Left out of this run after errors in a row. It takes part again in the next run."],
    offline: ["offline", "Not heard from: switched off, asleep or disconnected. Its layers moved to the others."],
    removed: ["removed", "Removed from the pool."],
  };

  function memory(d) {
    const t = d.telemetry || {};
    if (t.gpus && t.gpus.length) {
      const used = t.gpus.reduce((a, g) => a + (g.used_gb || 0), 0), total = t.gpus.reduce((a, g) => a + (g.total_gb || 0), 0);
      return { used, total, kind: `GPU memory${t.gpus.length > 1 ? ` (${t.gpus.length} GPUs)` : ""}`, extra: "" };
    }
    if (num(t.mlx_active_gb)) {
      return { used: t.ram_used_gb, total: t.ram_total_gb, kind: "memory (shared by CPU and GPU)",
        extra: `layerpool's layers and training here: ${gb(t.mlx_active_gb + (t.mlx_cache_gb || 0))}. The rest is the system and other apps.` };
    }
    return { used: t.ram_used_gb, total: t.ram_total_gb, kind: "RAM", extra: "" };
  }

  function card(d) {
    const t = d.telemetry || {};
    const mem = memory(d);
    const pct = mem.total ? Math.min(100, (100 * (mem.used || 0)) / mem.total) : 0;
    const capPct = d.cap_gb && mem.total ? Math.min(100, (100 * d.cap_gb) / mem.total) : null;
    const temps = (t.gpus || []).map((g) => g.temp_c).filter(num);
    const temp = num(t.temp_c) ? t.temp_c : temps.length ? Math.max(...temps) : null;
    const heat = num(temp) ? `${Math.round(temp)} °C` : t.heat ? esc(t.heat) : "not readable on this system";
    const [label, about] = statusText[d.status] || [d.status, ""];
    const last = state.history[state.history.length - 1];
    const tm = last && last.timing ? last.timing[d.name] : null;
    const total = tm ? (tm.compute + tm.wait + tm.send) || 1 : 1;
    const count = d.layers ? d.layers[1] - d.layers[0] : 0;
    const layers = d.layers ? `${d.layers[0]}–${d.layers[1] - 1} (${count} layers${num(d.on_gpu) && d.on_gpu < count ? `: ${d.on_gpu} on the GPU, ${count - d.on_gpu} in RAM` : ""})` : "none now";
    const memTip = `${mem.kind}: ${gb(mem.used)} in use of ${gb(mem.total)} (${pct.toFixed(0)}%).` + (mem.extra ? `\n${mem.extra}` : "") +
      (d.cap_gb ? `\nMemory cap: ${gb(d.cap_gb)} (the mark on the bar). layerpool gives this device only as many layers as fit under it.` : "\nNo cap set: layerpool uses what the device can spare.");
    const timeTip = tm ? `Where the last step's time went on this device:\ncompute ${fmt(tm.compute, 2)} s: running its layers\nwaiting ${fmt(tm.wait, 2)} s: for the device before it (or for your script)\nsending ${fmt(tm.send, 2)} s: passing results to the next device\nA lot of waiting or sending means the network or another device sets the pace.` : "";
    return `<div class="card" data-id="${d.id}">
      <div class="card-head"><input value="${esc(d.name)}" data-act="rename" title="Rename"><span class="pill ${d.status}" data-tip="${esc(about)}">${label}</span></div>
      <div class="hardware">${esc(d.devices.map((x) => x.name).join(", "))} · ${esc(d.backend || "")}</div>
      ${d.why_left_out ? `<div class="why">Last error: ${esc(d.why_left_out.slice(0, 240))}</div>` : ""}
      <div class="metric" data-tip="${esc(memTip)}">
        <div class="metric-head"><span>Memory</span><span>${gb(mem.used)} of ${gb(mem.total)} · ${pct.toFixed(0)}%</span></div>
        <div class="bar"><div class="fill${pct > 95 ? " full" : pct > 80 ? " high" : ""}" style="width:${pct}%"></div>${capPct !== null ? `<div class="capmark" style="left:${capPct}%"></div>` : ""}</div>
      </div>
      ${tm ? `<div class="metric" data-tip="${esc(timeTip)}">
        <div class="metric-head"><span>Last step</span><span>${fmt(tm.compute + tm.wait + tm.send, 1)} s</span></div>
        <div class="bar"><div class="c" style="flex:${tm.compute / total}"></div><div class="w" style="flex:${tm.wait / total}"></div><div class="s" style="flex:${tm.send / total}"></div></div>
        <div class="legend"><span><i style="background:var(--accent)"></i>compute ${fmt(tm.compute, 1)} s</span><span><i style="background:var(--wait)"></i>waiting ${fmt(tm.wait, 1)} s</span><span><i style="background:var(--send)"></i>sending ${fmt(tm.send, 1)} s</span></div>
      </div>` : ""}
      <div class="kv">
        <span>Layers</span><span>${layers}</span>
        <span data-tip="Seconds this device needs per layer and step, measured while training. Lower is faster; the split gives faster devices more layers.">Speed</span><span>${d.measured_s_per_layer ? fmt(d.measured_s_per_layer, 3) + " s per layer" : "measured once it trains"}</span>
        <span data-tip="Seconds this device spends per step passing results to the next device, measured while training. The split counts it, so a device on a slow link gets fewer layers.">Network</span><span>${num(d.link_s) ? fmt(d.link_s, 2) + " s sending per step" : "measured once it trains"}</span>
        <span>Temperature</span><span>${heat}</span>
        <span>Last seen</span><span>${ago(d.last_seen_s)}</span>
      </div>
      ${settingsBlock(d)}
      <div class="row">
        ${d.settings.enabled === false ? `<button class="small" data-act="enable">Use again</button>` : `<button class="small ghost" data-act="drain">Pause</button>`}
        <button class="small ghost danger" data-act="remove">Remove</button>
      </div>
    </div>`;
  }

  function settingsBlock(d) {
    const st = d.settings || {};
    const cuda = d.devices.filter((x) => x.kind === "cuda");
    const caps = st.gpu_caps || {};
    const gpuRows = cuda.length > 1 ? cuda.map((g, i) => `<div class="gpu-row">
        <label class="check" data-tip="Train on this GPU. Leave all ticked to use every GPU of the device."><input type="checkbox" data-gpu="${i}" ${!st.gpus || st.gpus.includes(i) ? "checked" : ""}> GPU ${i} · ${esc(g.name)} · ${gb(g.gb)}</label>
        <span class="field" data-tip="The most memory this GPU may use. Empty: its share of the device's cap, or what it can spare.">cap <input type="number" step="0.5" min="0.5" data-gpucap="${i}" value="${esc(caps[String(i)] ?? "")}" placeholder="auto"> GB</span>
      </div>`).join("") : "";
    const changed = ["layers", "gpus", "gpu_caps", "order", "cap_gb", "ram_cap"].some((k) => st[k] !== undefined && st[k] !== null);
    return `<details class="settings"${state.open[d.id] ? " open" : ""} data-id="${d.id}">
      <summary>Settings${changed ? ` <span class="muted small-text">(${esc(describe(st))})</span>` : ""}</summary>
      <div class="settings-grid">
        <span class="field" data-tip="${esc(capAbout(d))}">${cuda.length ? "GPU memory cap" : "Memory cap"} <input type="number" step="0.5" min="0.5" value="${esc(st.cap_gb ?? "")}" placeholder="auto" data-field="cap_gb"> GB</span>
        <span class="field" data-tip="How many of the model's layers this device takes: a count (8), a share of the model (30%) or a range (4-10). Empty: as many as suit its memory and speed.">Layers <input type="text" value="${esc(st.layers ?? "")}" placeholder="auto" data-field="layers"></span>
        <span class="field" data-tip="This device's place in the chain of devices: lower comes first. Empty: layerpool picks the order.">Order <input type="number" step="1" value="${esc(st.order ?? "")}" placeholder="auto" data-field="order"></span>
      </div>
      ${cuda.length ? `<div class="settings-grid"><span class="field" data-tip="Layers that do not fit the GPUs can also go to this machine's RAM, where its CPU runs them, more slowly. layerpool gives the RAM only as many layers as pay off. Empty: off.">Offload to RAM up to <input type="number" step="1" min="1" value="${esc(st.ram_cap ?? "")}" placeholder="off" data-field="ram_cap"> GB</span></div>` : ""}
      ${d.cap_gb && d.cap_gb !== st.cap_gb ? `<div class="preview">It ran out of memory, so it gets layers for at most ${gb(d.cap_gb)} until you set its cap.</div>` : ""}
      ${gpuRows}
      <div class="row"><button class="small primary" data-act="apply">Apply</button><button class="small ghost" data-act="reset" data-tip="Back to automatic for everything here.">Automatic</button></div>
    </details>`;
  }

  function capAbout(d) {
    const kind = d.devices[0] && d.devices[0].kind;
    const what = kind === "cuda" ? `GPU memory (VRAM)${d.devices.length > 1 ? " of the GPUs it trains on together, shared equally unless a GPU has its own cap below" : ""}. Its RAM is not counted.`
      : kind === "mlx" || kind === "mps" ? "the Mac's memory, which its CPU and GPU share."
      : "RAM, since it trains on its CPU.";
    const spare = kind === "cuda" ? "about 92% of each GPU" : kind === "mlx" || kind === "mps" ? "about 60% of the Mac's memory" : "about half its RAM";
    return `The most memory layerpool may use on this device: ${what}\nEmpty: what it can spare, ${spare}.`;
  }

  function describe(st) {
    const out = [];
    if (st.cap_gb) out.push(`cap ${st.cap_gb} GB`);
    if (st.layers) out.push(`layers ${st.layers}`);
    if (st.gpus) out.push(`GPUs ${st.gpus.join(",")}`);
    if (st.gpu_caps) out.push("GPU caps " + Object.entries(st.gpu_caps).map(([k, v]) => `${k}: ${v} GB`).join(", "));
    if (st.order !== undefined && st.order !== null) out.push(`order ${st.order}`);
    if (st.ram_cap) out.push(`up to ${st.ram_cap} GB in RAM`);
    return out.join(" · ");
  }

  function readSettings(box) {
    const value = (f) => box.querySelector(`[data-field=${f}]`).value.trim();
    const out = { cap_gb: value("cap_gb") ? parseFloat(value("cap_gb")) : null, layers: value("layers") || null,
      order: value("order") === "" ? null : parseInt(value("order"), 10) };
    const ram = box.querySelector("[data-field=ram_cap]");
    if (ram) out.ram_cap = ram.value.trim() ? parseFloat(ram.value) : null;
    const ticks = [...box.querySelectorAll("[data-gpu]")];
    if (ticks.length) {
      const on = ticks.filter((t) => t.checked).map((t) => parseInt(t.dataset.gpu, 10));
      if (!on.length) throw new Error("Tick at least one GPU (or pause the device instead).");
      out.gpus = on.length === ticks.length ? null : on;
      const caps = {};
      box.querySelectorAll("[data-gpucap]").forEach((c) => { if (c.value.trim()) caps[c.dataset.gpucap] = parseFloat(c.value); });
      out.gpu_caps = Object.keys(caps).length ? caps : null;
    }
    return out;
  }

  function renderDevices() {
    const list = state.devices.filter((d) => d.status !== "removed");
    $("device-count").textContent = list.length ? `${list.length}` : "";
    $("devices").innerHTML = list.length ? list.map(card).join("") : `<p class="muted">No devices yet. Use Add device to bring in any computer or notebook.</p>`;
  }

  function renderPending() {
    const rows = (state.status && state.status.waiting) || [];
    $("pending").innerHTML = rows.map((r) => `<div class="card waiting" data-code="${esc(r.code)}">
      <div class="card-head"><strong style="flex:1">${esc(r.name)}</strong><span class="pill" data-tip="This device ran layerpool join without a code and waits for you. Approve it only if the code it shows matches.">asks to join</span></div>
      <div class="hardware">${esc(r.hardware.join(", "))}${r.from ? ` · from ${esc(r.from)}` : ""} · asked ${ago(r.waiting_s)}</div>
      <div>Code on the device: <strong>${esc(r.code)}</strong></div>
      <div class="row"><button class="small primary" data-answer="yes">Approve</button><button class="small ghost danger" data-answer="no">Deny</button></div>
    </div>`).join("");
  }

  function renderEvents() {
    $("events").innerHTML = state.events.slice(-150).reverse().map((e) => {
      const cls = /error|left out|cannot|does not answer|failed|refused|denied|stopped answering/i.test(e.text) ? "bad"
        : /waiting|ran out|rolling back|overflow|skipped|note:/i.test(e.text) ? "warn" : "";
      return `<li class="${cls}"><time title="time since the pool started">${clock(e.t)}</time><span>${esc(e.text)}</span></li>`;
    }).join("");
  }

  async function refresh() {
    try {
      const [status, devices] = await Promise.all([api("/api/status"), api("/api/devices")]);
      state.status = status;
      state.devices = devices;
      state.history = status.history || [];
      status.history_len = state.history.length;
      state.events = (status.events || []).map(([t, text]) => ({ t, text }));
      renderRun();
      const focused = document.activeElement;
      if ((!focused || focused.tagName !== "INPUT" || !focused.closest(".card")) && !Object.keys(state.dirty).length) renderDevices();
      renderPending();
      renderEvents();
    } catch (e) {
      $("summary").textContent = e.status === 401 ? "open the dashboard link that `layerpool up` printed (the pool was restarted)" : "cannot reach the pool: " + e.message;
    }
  }

  $("pending").addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button[data-answer]");
    if (!btn) return;
    btn.disabled = true;
    try { await api("/api/answer", { code: btn.closest(".card").dataset.code, approve: btn.dataset.answer === "yes" }); } catch (e) { alert(e.message); }
    refresh();
  });

  $("devices").addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button[data-act]");
    if (!btn) return;
    const id = btn.closest(".card").dataset.id;
    const act = btn.dataset.act;
    try {
      if (act === "apply" || act === "reset") {
        const box = btn.closest(".settings");
        const pending = state.preview[id];
        const changes = pending ? pending.changes : act === "reset"
          ? { cap_gb: null, layers: null, order: null, ...(box.querySelector("[data-gpu]") ? { gpus: null, gpu_caps: null } : {}),
            ...(box.querySelector("[data-field=ram_cap]") ? { ram_cap: null } : {}) }
          : readSettings(box);
        if (!pending && state.status && state.status.run) {
          const res = await api("/api/preview", { [id]: changes });
          const text = res.plan ? "New split: " + res.plan.map((p) => `${p.name} ${p.range[0]}–${p.range[1] - 1}`).join(", ") + ". Press Apply again to confirm."
            : "No split fits these settings, so they would stop the run. Change them, or end the run first.";
          state.preview[id] = res.plan ? { changes } : null;
          const note = document.createElement("div");
          note.className = "preview";
          note.textContent = text;
          box.querySelectorAll(".preview.pending").forEach((n) => n.remove());
          note.classList.add("pending");
          box.querySelector(".row").before(note);
          if (!res.plan) delete state.preview[id];
          return;
        }
        delete state.preview[id];
        await api("/api/device", { id, ...changes });
        delete state.dirty[id];
      } else if (act === "drain") await api("/api/device/drain", { id });
      else if (act === "enable") await api("/api/device", { id, enabled: true });
      else if (act === "remove" && confirm("Remove this device from the pool? It will need a new invite to come back.")) await api("/api/device/remove", { id });
      refresh();
    } catch (e) { alert(e.message); }
  });

  $("devices").addEventListener("change", async (ev) => {
    if (ev.target.dataset.act !== "rename") return;
    const id = ev.target.closest(".card").dataset.id;
    try { await api("/api/device", { id, name: ev.target.value.trim() }); } catch (e) { alert(e.message); }
    refresh();
  });

  let invite = null, tab = "terminal";
  function showInvite() {
    $("invite-text").textContent = invite ? (tab === "terminal" ? invite.command : invite.notebook) : "";
    $("invite-validity").textContent = invite && invite.reusable
      ? `The code works for several devices and expires in ${Math.round(invite.expires_in_s / 3600)} hours.`
      : `The code works once and expires in ${invite ? Math.round(invite.expires_in_s / 3600) : 24} hours.`;
    $("invite-note").textContent = invite && !invite.tunnel
      ? "This pool has no tunnel, so only devices on this network can join. For notebooks (Colab, Kaggle) or devices elsewhere, start the pool with: layerpool up --tunnel"
      : "";
  }
  async function newInvite() { invite = await api("/api/invite", { reusable: $("invite-reusable").checked, ttl: parseFloat($("invite-hours").value) * 3600 }); showInvite(); }
  $("add-device").onclick = async () => { $("invite-text").textContent = "Creating a code…"; $("invite-dialog").showModal(); try { await newInvite(); } catch (e) { $("invite-text").textContent = e.message; } };
  $("invite-new").onclick = newInvite;
  $("invite-reusable").onchange = newInvite;
  $("invite-hours").onchange = newInvite;
  document.querySelectorAll(".tab").forEach((b) => b.onclick = () => { tab = b.dataset.tab; document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === b)); showInvite(); });
  $("invite-copy").onclick = () => navigator.clipboard.writeText($("invite-text").textContent);

  $("devices").addEventListener("toggle", (ev) => {
    const box = ev.target.closest && ev.target.closest(".settings");
    if (box) {
      state.open[box.dataset.id] = box.open;
      if (!box.open) { delete state.dirty[box.dataset.id]; delete state.preview[box.dataset.id]; }
    }
  }, true);
  $("devices").addEventListener("input", (ev) => {
    const box = ev.target.closest(".settings");
    if (box) { state.dirty[box.dataset.id] = true; delete state.preview[box.dataset.id]; box.querySelectorAll(".preview.pending").forEach((n) => n.remove()); }
  });
  $("rebalance").onchange = async () => {
    try { await api("/api/run/settings", { rebalance: $("rebalance").checked }); } catch (e) { alert(e.message); }
    refresh();
  };
  $("resplit").onclick = async () => {
    try { await api("/api/run/settings", { resplit: true }); } catch (e) { alert(e.message); }
    refresh();
  };

  $("end-run").onclick = async () => {
    if (!confirm("End this run? Unsaved progress since the last save is dropped.")) return;
    try { await api("/api/run/end", { state: "stopped" }); } catch (e) { alert(e.message); }
    refresh();
  };

  let resizing = null;
  window.addEventListener("resize", () => { clearTimeout(resizing); resizing = setTimeout(() => charts.forEach((c) => chart($(c.el), c, true)), 150); });

  function stream() {
    const es = new EventSource("/api/events?t=" + encodeURIComponent(token));
    es.addEventListener("step", (e) => { state.history.push(JSON.parse(e.data)); renderRun(); });
    es.addEventListener("event", (e) => { state.events.push(JSON.parse(e.data)); renderEvents(); });
    es.addEventListener("plan", () => refresh());
    es.addEventListener("metric", (e) => { state.metrics = JSON.parse(e.data); renderRun(); });
    es.onerror = () => { es.close(); setTimeout(stream, 3000); };
  }
  refresh();
  stream();
  setInterval(refresh, 2500);
})();
