/* agent-router demo UI: vanilla JS, no build. One checkpoint renderer (renderCheckpoint)
   is shared by the Playground, the Live timeline, the backend comparison and Replay. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  catalog: null, // /api/catalog
  entries: new Map(), // id -> entry
  backends: [], // [{name, available}]
  threshold: 0.5,
  thrTouched: false, // until the slider moves, each backend routes at its own calibrated threshold
};

const PRESETS = [
  { label: "17% of 2,340", point: "prompt", text: "What is 17% of 2,340 exactly?" },
  {
    label: "python3 -c 2**200", point: "tool", tool: "Bash",
    text: "compute 2**200 exactly", input: { command: 'python3 -c "print(2**200)"' },
  },
  {
    label: "Read release.html", point: "tool", tool: "Read",
    text: "summarize the release notes page", input: { file_path: "docs/release.html" },
  },
  {
    label: "failed orders", point: "prompt",
    text: "List the ids of orders whose status is failed in data/orders.json",
  },
  {
    label: "release-notes skill", point: "skill", tool: "Skill",
    text: "Write a commit message for adding retry logic to the HTTP client",
    input: { skill: "release-notes" },
  },
  { label: "run the test suite", point: "prompt", text: "run the test suite" },
];
const POINT_NAMES = { prompt: "Prompt", tool: "Tool", skill: "Skill" };
const POINT_LONG = { prompt: "User prompt", tool: "Tool call", skill: "Skill call" };

/* -- small helpers -------------------------------------------------------- */

function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

const pct = (p) => `${(p * 100).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%`;
const ms = (v) => (v == null ? "" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) {
    const detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText;
    throw new Error(detail);
  }
  return body;
}

function optionMeta(id) {
  if (id === "none") return state.catalog ? state.catalog.none : { id: "none", name: "Native path" };
  return state.entries.get(id) || { id, name: id };
}

/* -- the checkpoint model -------------------------------------------------- */

/** Normalise a route response, a live `decision` event or an audit record. */
function toCheckpoint(src) {
  const d = src.decision || src; // route response nests decision; events/records are flat
  const result = src.result !== undefined ? src.result : null;
  const probabilities = (result && result.probabilities) || src.probabilities || {};
  const choice = (result && result.choice) || src.choice || null;
  let optionIds = src.options && src.options.length
    ? src.options.map((o) => (typeof o === "string" ? o : o.id))
    : Object.keys(probabilities);
  if (!optionIds.includes("none") && Object.keys(probabilities).length) optionIds.push("none");
  optionIds = [...optionIds.filter((i) => i !== "none"), ...(optionIds.includes("none") ? ["none"] : [])];
  // The bar the chosen entry was held to: per-entry thresholds differ from the router's.
  // Records written before `applied_threshold` existed still name it in their reason.
  const reasonThr = /threshold (\d*\.\d+)/.exec(d.reason || "");
  const entryThr = d.entry_id && state.entries.get(d.entry_id) ? state.entries.get(d.entry_id).threshold : null;
  const thr = d.applied_threshold ?? (reasonThr ? Number(reasonThr[1]) : null) ?? entryThr
    ?? src.threshold ?? (src.thresholds && src.thresholds.threshold) ?? state.threshold;
  return {
    point: src.point || (src.event && src.event.point) || "prompt",
    tool: src.tool_name || null,
    action: d.action,
    reason: d.reason || null,
    entryId: d.entry_id || null,
    choice,
    probabilities,
    optionIds,
    confidence: (result && result.confidence) ?? src.confidence ?? null,
    backend: (result && result.backend) || src.backend || null,
    stages: (result && result.stages) || src.stages || [],
    latency: src.latency_ms ?? (result && result.latency_ms) ?? null,
    hint: src.hint || null,
    hintRestored: !!src.hint_restored,
    threshold: thr,
    mode: src.mode || (src.thresholds && src.thresholds.mode) || null,
    state: src.state || null,
    text: src.text || null,
  };
}

/** Visual state: suggest | enforce | gated (pointed, but below threshold) | native | skipped | error */
function tone(cp) {
  if (cp.action === "suggest") return "suggest";
  if (cp.action === "enforce") return "enforce";
  if (cp.action === "native" && cp.entryId) return "gated";
  if (cp.action === "native" && !Object.keys(cp.probabilities).length && cp.reason && cp.reason.startsWith("decider error")) return "error";
  if (cp.action === "skipped") return "skipped";
  return "native";
}

function verdict(cp) {
  const t = tone(cp);
  const name = cp.entryId ? optionMeta(cp.entryId).name : null;
  const p = cp.entryId ? cp.probabilities[cp.entryId] : null;
  switch (t) {
    case "suggest": return { title: `Points to ${name}`, pill: `Hint injected: ${vsThreshold(p, cp.threshold)}` };
    case "enforce": return { title: `Blocks ${cp.tool || "the call"}, points to ${name}`, pill: `Call denied: ${vsThreshold(p, cp.threshold)}` };
    case "gated": return { title: `Leans to ${name}, not enough`, pill: vsThreshold(p, cp.threshold) };
    case "error": return { title: "Classifier failed, agent continues", pill: "Fail open" };
    case "skipped": return { title: skipTitle(cp.reason), pill: "Skipped" };
    default: return { title: "None fits, agent's own tools", pill: "Native path" };
  }
}

/** "69.6% is below the 80.0% threshold": which side of the bar the chosen option landed on. */
function vsThreshold(p, thr) {
  if (p == null || thr == null) return "";
  const side = p > thr ? "above" : p < thr ? "below" : "at";
  // one more decimal when rounding would print the same figure on both sides
  const fmt = (x) => (side !== "at" && pct(p) === pct(thr) ? `${(x * 100).toFixed(2)}%` : pct(x));
  return `${fmt(p)} is ${side} the ${fmt(thr)} threshold`;
}

function skipTitle(reason) {
  if (!reason) return "Not routed";
  if (reason.includes("own tool")) return "Already a catalog tool, not routed";
  if (reason.includes("no eligible")) return "No catalog option covers this tool";
  if (reason.includes("already suggested")) return "Already hinted this turn";
  if (reason.includes("disabled")) return "Router disabled";
  return reason;
}

/** Plain text with **bold** spans (no HTML is ever interpreted). */
function richText(text) {
  const frag = document.createDocumentFragment();
  String(text).split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (/^\*\*[^*]+\*\*$/.test(part)) frag.append(el("b", { text: part.slice(2, -2) }));
    else if (part) frag.append(part);
  });
  return frag;
}

/* -- rendering ------------------------------------------------------------- */

/** Bars for one distribution. `showThr=false` for a cascade stage that did not decide. */
function renderBars(cp, t, showThr = true) {
  const bars = el("div", { class: "bars", role: "list", "aria-label": "Probability per option" });
  const thr = Math.max(0, Math.min(1, cp.threshold));
  const edge = thr < 0.08 ? "edge-l" : thr > 0.92 ? "edge-r" : "";
  if (showThr) {
    bars.append(
      el("span"),
      el("div", { class: "thr-head", "aria-hidden": "true" },
        el("span", { class: `thr-tag ${edge}`, style: `left:${thr * 100}%`, text: `threshold ${pct(thr)}` })),
      el("span"),
    );
  }
  cp.optionIds.forEach((id) => {
    const isNone = id === "none";
    if (isNone && cp.optionIds.length > 1) bars.append(el("div", { class: "bar-sep", "aria-hidden": "true" }));
    const p = cp.probabilities[id] ?? 0;
    const chosen = id === cp.choice;
    const meta = optionMeta(id);
    const fill = el("span", { class: "bar-fill" });
    const row = el("div", {
      class: `bar-row${chosen ? ` chosen ${t}` : ""}${isNone ? " is-none" : ""}`,
      role: "listitem",
      "aria-label": `${meta.name}: ${pct(p)}${chosen ? ", chosen" : ""}${chosen && showThr && !isNone ? `, ${vsThreshold(p, thr).replace(/^\S+ is /, "")}` : ""}`,
    },
      el("div", { class: "bar-label", title: meta.what || "" },
        el("span", { class: "nm", text: isNone ? "none" : meta.name }),
        el("span", { class: "id", text: isNone ? "the agent's own tools" : id })),
      el("div", { class: "bar-track" }, fill,
        showThr ? el("span", { class: "thr-mark", style: `left:${thr * 100}%`, "aria-hidden": "true" }) : null),
      el("div", { class: "bar-val", text: Object.keys(cp.probabilities).length ? pct(p) : "–" }),
    );
    bars.append(row);
    requestAnimationFrame(() => { fill.style.width = `${Math.max(p * 100, p > 0 ? 0.6 : 0)}%`; });
  });
  return bars;
}

/* -- cascade stages (local classifier, then Jev) --------------------------------- */

const STAGE_NAMES = { primary: "Local classifier (MIT, offline)", confirm: "Jev (confirms)" };

function cascadeNote(cp) {
  if (cp.backend === "cascade:local") return { cls: "local", text: "Answered locally", why: "The local classifier was confidently none; Jev was not asked." };
  if (cp.backend === "cascade:local-fallback") return { cls: "fallback", text: "Jev failed: local fallback", why: "Escalated, but Jev failed: the local answer is used, biased to none." };
  return { cls: "escalated", text: "Escalated to Jev", why: "The local answer was not a confident none, so Jev confirms." };
}

/** One labelled bar set per stage; only the deciding one shows the router threshold. */
function renderStages(cp, t) {
  const note = cascadeNote(cp);
  const box = el("div", { class: "stages" }, el("p", { class: `stage-note ${note.cls}`, text: note.text, title: note.why }));
  const fallback = cp.backend === "cascade:local-fallback";
  cp.stages.forEach((st, i) => {
    const deciding = !fallback && i === cp.stages.length - 1;
    const head = el("div", { class: "stage-head" },
      el("span", { class: "stage-name", text: STAGE_NAMES[st.role] || st.role }),
      el("span", { class: "stage-meta", text: [st.backend, st.latency_ms != null ? ms(st.latency_ms) : null].filter(Boolean).join(", ") }));
    const body = st.skipped
      ? el("p", { class: "cmp-off", text: "Skipped: Jev failed repeatedly, so it is paused for a minute (circuit open)." })
      : st.failed || st.error_type
      ? el("p", { class: "cmp-off", text: `Failed (${st.error_type || "error"}). Details are in the audit log.` })
      : renderBars({ ...cp, probabilities: st.probabilities || {}, choice: st.choice }, deciding ? t : "stage", deciding);
    box.append(el("section", { class: `stage${deciding ? " deciding" : ""}` }, head, body));
  });
  if (fallback) {
    box.append(el("section", { class: "stage deciding" },
      el("div", { class: "stage-head" }, el("span", { class: "stage-name", text: "Fallback answer" })),
      renderBars(cp, t)));
  }
  return box;
}

function renderSees(cp, t) {
  const box = el("div", { class: "sees" });
  if (cp.hint) {
    box.append(
      el("h4", { text: t === "enforce" ? `The ${cp.tool || "native"} call is denied with this reason` : "What the agent sees (added to its context)" }),
      el("pre", { class: `hint-text ${t}`, text: cp.hint }),
    );
    if (cp.hintRestored) box.append(el("p", { class: "note", text: "The audit log keeps 300 characters; the rest is re-rendered from the same catalog template." }));
  } else {
    const why = t === "gated"
      ? `Nothing. ${cp.entryId ? optionMeta(cp.entryId).name : "The top option"} is at ${vsThreshold(cp.probabilities[cp.entryId], cp.threshold).replace(/ is /, ", ")}, so the hook returns {} and the agent carries on unchanged.`
      : t === "error"
        ? `Nothing. ${cp.reason || "The classifier raised"}; the router fails open.`
        : "Nothing. The hook returns {} and the agent carries on unchanged.";
    box.append(el("h4", { text: "What the agent sees" }), el("p", { class: "nothing", text: why }));
  }
  return box;
}

function renderDetails(cp) {
  const dl = el("dl");
  const add = (k, v) => { if (v != null && v !== "") dl.append(el("dt", { text: k }), el("dd", { text: v })); };
  add("Router reason", cp.reason);
  add("Confidence", cp.confidence != null ? cp.confidence.toFixed(3) : null);
  add("Mode", cp.mode);
  const d = el("details", { class: "why" }, el("summary", { text: "Why this decision" }), dl);
  if (cp.state) d.append(el("pre", { text: cp.state }));
  return d;
}

/** Fill `root` with one checkpoint. opts: {compact, title, extra} */
function renderCheckpoint(root, src, opts = {}) {
  const cp = toCheckpoint(src);
  const t = tone(cp);
  const v = verdict(cp);
  root.replaceChildren();
  root.classList.toggle("compact", !!opts.compact);
  const where = el("div", { class: "cp-where" },
    opts.title ? opts.title : `${POINT_LONG[cp.point] || cp.point} checkpoint`,
    cp.tool ? " on " : "", cp.tool ? el("span", { class: "tool", text: cp.tool }) : "");
  const meta = el("div", { class: "cp-meta" },
    cp.backend ? `${cp.backend}` : "", cp.latency != null ? `, ${ms(cp.latency)}` : "");
  root.append(
    el("div", { class: "cp-head" }, where, meta),
    el("div", { class: "verdict" }, el("h3", { text: v.title }), el("span", { class: `pill ${t}`, text: v.pill })),
  );
  if (cp.stages.length) root.append(renderStages(cp, t));
  else if (cp.optionIds.length) root.append(renderBars(cp, t));
  if (opts.compact) {
    if (opts.extra) root.append(opts.extra);
    if (t === "error" && cp.reason) root.append(el("p", { class: "cmp-off", text: cp.reason }));
    return cp;
  }
  root.append(renderSees(cp, t), renderDetails(cp));
  return cp;
}

/* -- playground ------------------------------------------------------------- */

const form = $("#step-form");
let routeSeq = 0;

function currentPoint() { return $("input[name=point]:checked", form).value; }

function syncPointFields() {
  const point = currentPoint();
  $("#tool-fields").hidden = point === "prompt";
  $("#text-label").textContent = point === "prompt" ? "Prompt" : "Prompt the agent is working on";
  const toolSel = $("#tool-name");
  if (point === "skill") { ensureOption(toolSel, "Skill"); toolSel.value = "Skill"; toolSel.disabled = true; }
  else { toolSel.disabled = false; if (toolSel.value === "Skill") toolSel.value = "Bash"; }
}

function ensureOption(sel, value) {
  if (![...sel.options].some((o) => o.value === value)) sel.append(el("option", { text: value }));
}

function applyPreset(p, btn) {
  $$(".preset").forEach((b) => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
  $(`input[name=point][value=${p.point}]`, form).checked = true;
  $("#text").value = p.text;
  if (p.tool) { ensureOption($("#tool-name"), p.tool); $("#tool-name").value = p.tool; }
  $("#tool-input").value = p.input ? JSON.stringify(p.input, null, 2) : "";
  syncPointFields();
  runRoute();
}

function readStep() {
  const point = currentPoint();
  const step = { text: $("#text").value, point };
  if (point !== "prompt") {
    step.tool_name = $("#tool-name").value;
    const raw = $("#tool-input").value.trim();
    if (raw) {
      try { step.tool_input = JSON.parse(raw); } catch { throw new Error("Tool input must be a JSON object, for example {\"command\": \"ls\"}."); }
      if (typeof step.tool_input !== "object" || Array.isArray(step.tool_input)) throw new Error("Tool input must be a JSON object.");
    }
  }
  step.mode = $("input[name=mode]:checked", form).value;
  // untouched slider: the server applies the backend's own (calibrated) threshold
  step.threshold = state.thrTouched ? Number($("#threshold").value) : null;
  return step;
}

/** Show the threshold the server used while the slider is untouched (backend default). */
function showThreshold(thr) {
  if (thr == null) return;
  $("#threshold").value = thr;
  $("#thr-out").textContent = `${Number(thr).toFixed(2)} auto`;
  $("#thr-out").title = "The classifier's own calibrated threshold. Move the slider to override it.";
}

function showFormError(msg) {
  const e = $("#form-error");
  e.hidden = !msg;
  e.textContent = msg || "";
}

async function route(step, backend) {
  return api("/api/route", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...step, backend }),
  });
}

async function runRoute() {
  let step;
  try { step = readStep(); } catch (err) { showFormError(err.message); return; }
  showFormError("");
  const seq = ++routeSeq;
  const board = $("#board");
  board.classList.add("loading");
  try {
    const res = await route(step, $("#backend").value || "local");
    if (seq !== routeSeq) return;
    if (!state.thrTouched) showThreshold(res.threshold);
    renderCheckpoint(board, res);
  } catch (err) {
    if (seq === routeSeq) showFormError(`Routing failed: ${err.message}`);
  } finally {
    if (seq === routeSeq) board.classList.remove("loading");
  }
  if (!$("#compare").hidden) runCompare();
}

async function runCompare() {
  let step;
  try { step = readStep(); } catch (err) { showFormError(err.message); return; }
  const section = $("#compare");
  const grid = $("#compare-grid");
  section.hidden = false;
  grid.replaceChildren();
  for (const b of state.backends) {
    const card = el("article", { class: "checkpoint compact", "aria-live": "polite" });
    grid.append(card);
    if (!b.available) {
      card.append(el("div", { class: "cmp-name", text: b.name }),
        el("p", { class: "cmp-off", text: "Not available here: install its extra or set its API key." }));
      continue;
    }
    card.append(el("div", { class: "cmp-name", text: b.name }), el("p", { class: "cmp-off", text: "Asking…" }));
    const t0 = performance.now();
    route(step, b.name).then((res) => {
      renderCheckpoint(card, res, { compact: true, title: el("span", { class: "cmp-name", text: b.name }) });
      $(".cp-meta", card).replaceChildren(el("span", { class: "latency", text: ms(res.latency_ms) }));
    }).catch((err) => {
      card.replaceChildren(el("div", { class: "cmp-name", text: b.name }),
        el("p", { class: "cmp-off", text: `Failed after ${ms(performance.now() - t0)}: ${err.message}` }));
    });
  }
}

function initPlayground() {
  const presets = $("#presets");
  PRESETS.forEach((p) => {
    const btn = el("button", { type: "button", class: "preset", "aria-pressed": "false" },
      el("span", { class: "pt", text: POINT_NAMES[p.point] }), p.label);
    btn.addEventListener("click", () => applyPreset(p, btn));
    presets.append(btn);
  });
  $$("input[name=point]", form).forEach((r) => r.addEventListener("change", syncPointFields));
  form.addEventListener("submit", (e) => { e.preventDefault(); runRoute(); });
  $("#compare-btn").addEventListener("click", runCompare);
  let slideTimer;
  $("#threshold").addEventListener("input", (e) => {
    state.thrTouched = true;
    $("#thr-out").textContent = Number(e.target.value).toFixed(2);
    clearTimeout(slideTimer);
    slideTimer = setTimeout(runRoute, 120);
  });
  $$("input[name=mode]", form).forEach((r) => r.addEventListener("change", runRoute));
  $("#backend").addEventListener("change", runRoute);
  applyPreset(PRESETS[0], $(".preset"));
}

/* -- live agent -------------------------------------------------------------- */

let live = null; // EventSource

function tlItem(kind, cls, ...body) {
  return tlInsert(null, kind, cls, ...body);
}

/** Add a timeline item, before `before` when given (else at the end). */
function tlInsert(before, kind, cls, ...body) {
  const li = el("li", { class: `tl ${cls}` }, el("div", { class: "tl-kind", text: kind }), el("div", { class: "tl-body" }, ...body));
  $("#timeline").insertBefore(li, before);
  li.scrollIntoView({ block: "nearest", behavior: "smooth" });
  return li;
}

function liveStatus(...kids) {
  const s = $("#live-status");
  s.hidden = false;
  s.replaceChildren(...kids);
}

function stopLive(message) {
  if (live) { live.close(); live = null; }
  $("#live-run").disabled = false;
  $("#live-stop").hidden = true;
  $$(".tl.pending").forEach((n) => n.remove());
  if (message) liveStatus(message);
}

async function startLive(e) {
  e.preventDefault();
  const prompt = $("#live-prompt").value.trim();
  if (!prompt) { liveStatus("Type a prompt for the agent first."); return; }
  stopLive();
  $("#timeline").replaceChildren();
  const mode = $("input[name=live-mode]:checked").value;
  const backend = $("#live-backend").value || "local";
  $("#live-run").disabled = true;
  liveStatus(`Starting a ${mode} session with the ${backend} classifier…`);
  let start;
  try {
    // POST mints a one-time token; the stream itself is a plain same-origin GET.
    start = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, mode, backend }),
    });
  } catch (err) {
    $("#live-run").disabled = false;
    liveStatus(`The server did not start the run: ${err.message}`);
    return;
  }
  $("#live-stop").hidden = false;
  const toolItems = new Map();
  let lastCheckpoint = null;
  let session = null;
  const pending = () => { $$(".tl.pending").forEach((n) => n.remove()); tlItem("", "pending", "Agent is working…"); };

  live = new EventSource(start.stream);
  const on = (type, fn) => live.addEventListener(type, (ev) => fn(JSON.parse(ev.data)));
  on("session", (d) => {
    session = d.session;
    state.threshold = d.threshold;
    liveStatus(`Session ${d.session}, ${d.mode} mode, threshold ${d.threshold.toFixed(2)}, ${d.backend} classifier.`);
  });
  on("prompt", (d) => { tlItem("You asked", "prompt", d.prompt); pending(); });
  on("decision", (d) => {
    $$(".tl.pending").forEach((n) => n.remove());
    const kind = d.point === "prompt" ? "Checkpoint: user prompt" : `Checkpoint: before ${d.tool_name}`;
    // The SDK streams the tool_use block before its PreToolUse hook runs: put the
    // checkpoint above the call it guards.
    const guarded = d.tool_name ? [...toolItems.values()].reverse()
      .find((li) => li.dataset.tool === d.tool_name && !li.dataset.checked) : null;
    if (guarded) guarded.dataset.checked = "1";
    const before = guarded || null;
    let li;
    if (d.action === "skipped" && !Object.keys(d.probabilities || {}).length) {
      // gated out before the classifier was asked: one quiet line
      const why = (d.tool_name || "").startsWith("mcp__agent_router__")
        ? "already a catalog tool, not routed" : "no catalog option covers this tool, not routed";
      li = tlInsert(before, kind, "cp skipped", el("span", { class: "muted", text: `Skipped: ${why}.` }));
    } else {
      const card = el("article", { class: "checkpoint" });
      const cp = renderCheckpoint(card, { ...d, threshold: d.applied_threshold ?? state.threshold, mode }, { compact: true });
      const t = tone(cp);
      if (cp.hint) card.append(el("pre", { class: `hint-text ${t}`, text: cp.hint }));
      li = tlInsert(before, kind, `cp ${t}`, card);
    }
    lastCheckpoint = li;
    pending();
  });
  on("hook", (d) => {
    if (!lastCheckpoint) return;
    const txt = { additionalContext: "Hook returned additionalContext: the hint above joins the agent's context.",
      deny: "Hook returned deny: the native call does not run.", none: "Hook returned {}: nothing changes." }[d.output] || d.output;
    $(".tl-body", lastCheckpoint).append(el("div", { class: "hook-out", text: txt }));
  });
  on("tool_use", (d) => {
    $$(".tl.pending").forEach((n) => n.remove());
    const li = tlItem("Tool call", "tool", el("span", { class: "tool-name", text: d.name }),
      el("pre", { text: JSON.stringify(d.input, null, 2) }));
    li.dataset.tool = d.name;
    toolItems.set(d.id, li);
    pending();
  });
  on("tool_result", (d) => {
    const li = toolItems.get(d.tool_use_id);
    const det = el("details", {}, el("summary", { text: d.is_error ? "Result (error)" : "Result" }), el("pre", { text: d.content }));
    if (li) $(".tl-body", li).append(det);
    else tlItem("Tool result", "tool", det);
  });
  on("assistant", (d) => {
    $$(".tl.pending").forEach((n) => n.remove());
    tlItem("Agent", "say", richText(d.text)).dataset.text = d.text;
    pending();
  });
  on("result", (d) => {
    $$(".tl.pending").forEach((n) => n.remove());
    const said = $$(".tl.say").pop();
    if (said && d.result && said.dataset.text.trim() === d.result.trim()) said.remove(); // shown as the answer
    const cost = [d.num_turns != null ? `${d.num_turns} turns` : null, d.duration_ms != null ? ms(d.duration_ms) : null,
      d.total_cost_usd != null ? `$${d.total_cost_usd.toFixed(4)}` : null].filter(Boolean).join(", ");
    tlItem(d.is_error ? "Finished with an error" : "Answer", d.is_error ? "answer err" : "answer", richText(d.result || "(no text)"), cost ? el("div", { class: "cost", text: cost }) : null);
  });
  live.addEventListener("error", (ev) => {
    if (ev.data) { // a server-sent `error` event: the run failed
      const d = JSON.parse(ev.data);
      tlItem("Run failed", "err", d.message || "unknown error");
      return;
    }
    // a transport error: close now so EventSource never reconnects (that would start a new run)
    stopLive(live && live.readyState === EventSource.CLOSED ? "Connection closed." : "Lost the connection to the demo server.");
  });
  on("done", () => {
    stopLive();
    if (session) {
      liveStatus(`Session ${session} finished. `, el("a", { href: "#replay", onclick: (ev) => { ev.preventDefault(); openReplay(session); }, text: "Replay its checkpoints" }));
    }
  });
}

function initLive() {
  $("#live-form").addEventListener("submit", startLive);
  $("#live-stop").addEventListener("click", () => stopLive("Stopped. The server cancels the agent run."));
}

/* -- replay ------------------------------------------------------------------ */

const replay = { records: [], index: 0 };

async function loadSessions(select) {
  const sel = $("#session-select");
  let sessions = [];
  try { sessions = (await api("/api/audit/sessions")).sessions; } catch { sessions = []; }
  sel.replaceChildren(...sessions.map((s) => el("option", { value: s.session },
    `${s.session}${s.sample ? " (sample)" : ""}: ${s.records} checkpoint${s.records === 1 ? "" : "s"}${s.first_text ? `, “${s.first_text.slice(0, 48)}”` : ""}`)));
  $("#replay-empty").hidden = sessions.length > 0;
  $("#replay-body").hidden = sessions.length === 0;
  if (!sessions.length) return;
  sel.value = select && sessions.some((s) => s.session === select) ? select : sessions[0].session;
  await loadSession(sel.value);
}

async function loadSession(name) {
  const data = await api(`/api/audit/${encodeURIComponent(name)}`);
  replay.records = data.records;
  replay.index = 0;
  const pills = $("#step-pills");
  pills.replaceChildren(...replay.records.map((r, i) => {
    const t = tone(toCheckpoint(r));
    const label = r.point === "prompt" ? "P" : r.point === "skill" ? "S" : "T";
    return el("li", {}, el("button", { type: "button", class: t, title: `${r.point}${r.tool_name ? ` ${r.tool_name}` : ""}${r.agent_type ? ` (inside ${r.agent_type})` : ""}: ${r.action}`,
      "aria-label": `Checkpoint ${i + 1}, ${r.point}, ${r.action}`, onclick: () => showStep(i) }, `${i + 1}${label}`));
  }));
  showStep(0);
}

function showStep(i) {
  const n = replay.records.length;
  if (!n) return;
  replay.index = Math.max(0, Math.min(n - 1, i));
  const rec = replay.records[replay.index];
  renderCheckpoint($("#replay-board"), rec, { title: `Checkpoint ${replay.index + 1} of ${n}: ${POINT_LONG[rec.point] || rec.point}` });
  const board = $("#replay-board");
  if (rec.text) board.insertBefore(el("p", { class: "muted", text: `Prompt: ${rec.text}` }), board.children[1]);
  board.insertBefore(el("p", { class: "muted", text: rec.agent_type ? `Called inside the ${rec.agent_type} agent` : "Called on the main thread" }), board.children[1]);
  $$("#step-pills button").forEach((b, j) => (j === replay.index ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
  $("#step-prev").disabled = replay.index === 0;
  $("#step-next").disabled = replay.index === n - 1;
}

function openReplay(session) {
  selectTab("replay");
  loadSessions(session);
}

function initReplay() {
  $("#session-select").addEventListener("change", (e) => loadSession(e.target.value));
  $("#session-refresh").addEventListener("click", () => loadSessions($("#session-select").value));
  $("#step-prev").addEventListener("click", () => showStep(replay.index - 1));
  $("#step-next").addEventListener("click", () => showStep(replay.index + 1));
  document.addEventListener("keydown", (e) => {
    if ($("#view-replay").hidden || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
    if (e.key === "ArrowLeft") showStep(replay.index - 1);
    if (e.key === "ArrowRight") showStep(replay.index + 1);
  });
}

/* -- trace ------------------------------------------------------------------- */
/* Decision traces from an integration's record-only hooks (integrations/first-principles/
   trace.py): what the agent did, what it decided, and where the router's notes landed. */

const trace = { data: null, runs: [], index: 0 };
const BAND_TONE = { Rigorous: "suggest", Sound: "gated", "Hand-wavy": "enforce", Absent: "enforce" };
const CONF_TONE = { HIGH: "suggest", MEDIUM: "gated", LOW: "enforce" };
const VERDICT_TONE = { Accept: "native", Challenge: "gated", Discard: "enforce" };
const REPORT_OP = { create: "Creates the report file", revise: "Revises the report in place", check: "Checks the report" };

const chip = (text, tone = "native", title) => el("span", { class: `chip ${tone}`, title, text });
const when = (ts) => Date.parse(ts);
const clock = (s) => (s == null ? "" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);
/** The calculator's result text, unwrapped from its JSON envelope when it has one. */
function calcResult(text) {
  try {
    const v = JSON.parse(text);
    if (v && typeof v === "object") return v.result ?? v.value ?? text;
  } catch { /* plain text */ }
  return text;
}
/** A multi-line command as one short line: `python3 -c "` alone says nothing. */
const oneLine = (cmd, n = 90) => {
  const flat = String(cmd || "").replace(/\s*\n\s*/g, " ⏎ ").trim();
  return flat.length > n ? `${flat.slice(0, n - 1)}…` : flat;
};
const minutes = (s) => (s == null ? "–" : s >= 90 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`);

async function loadTraceSessions(select) {
  let body = { sessions: [] };
  try { body = await api("/api/trace/sessions"); } catch { /* shown as empty */ }
  const sel = $("#trace-select");
  sel.replaceChildren(...body.sessions.map((s) => el("option", { value: s.session },
    `${s.session.slice(0, 8)}: ${s.finished} of ${s.runs} run${s.runs === 1 ? "" : "s"} finished${s.title ? `, “${s.title.slice(0, 70)}”` : ""}`)));
  const none = body.sessions.length === 0;
  $("#trace-empty").hidden = !none;
  $("#trace-body").hidden = none;
  if (none) {
    $("#trace-empty").replaceChildren(
      el("p", { text: "No decision traces yet." }),
      el("p", { class: "small-print", text: `Looking in ${body.trace_dir || "the trace folder"}. The first-principles integration's hooks write one trace per session; trace.py --replay makes one from a saved run.` }));
    return;
  }
  sel.value = select && body.sessions.some((s) => s.session === select) ? select : body.sessions[0].session;
  await loadTrace(sel.value);
}

async function loadTrace(session) {
  const data = await api(`/api/trace/${encodeURIComponent(session)}`);
  trace.data = data;
  const byRun = new Map();
  let handoff = []; // the main session's Agent call(s) that start the next run
  for (const r of data.records) {
    if (r.kind === "delegation" && !r.agent_id) { handoff.push(r); continue; }
    const key = r.agent_id || "run";
    if (!byRun.has(key)) { byRun.set(key, handoff); handoff = []; }
    byRun.get(key).push(r);
  }
  trace.runs = [...byRun.entries()].map(([id, recs]) => ({
    id, recs, start: recs.find((r) => r.kind === "run_start"), end: recs.find((r) => r.kind === "run_end"),
  }));
  const pills = $("#run-pills");
  pills.hidden = trace.runs.length < 2;
  pills.replaceChildren(...trace.runs.map((r, i) => el("li", {}, el("button", {
    type: "button", class: r.end ? "suggest" : "gated", onclick: () => showRun(i),
    "aria-label": `Run ${i + 1}${r.end ? "" : ", not finished"}`,
  }, `Run ${i + 1}`))));
  showRun(0);
}

/** The router's notes that fired inside this run (or, for the prompt, just before it). */
function routerMarks(run) {
  const start = run.start ? when(run.start.ts) : -Infinity;
  const end = run.end ? when(run.end.ts) : Infinity;
  return (trace.data.audit || []).filter((a) => a.agent_type && ["suggest", "enforce"].includes(a.action)
    && when(a.ts) >= start && when(a.ts) <= end);
}

function appliedThreshold(rec) {
  if (rec.applied_threshold != null) return rec.applied_threshold;
  const m = /threshold (\d*\.\d+)/.exec(rec.reason || "");
  return m ? Number(m[1]) : null;
}

function fixRepeat(d) {
  if (d.gate && d.gate.fix_repeat) return "yes";
  const edge = d.re_entry || {};
  if (edge.fired && /fix\/repeat/i.test(edge.disclosure || "")) return "yes (disclosed)";
  return "no";
}

function showRun(i) {
  trace.index = i;
  const run = trace.runs[i];
  $$("#run-pills button").forEach((b, j) => (j === i ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
  renderTraceSummary(run);
  renderTraceTimeline(run);
  renderDecisions(run);
}

/** Calculator notes and calls, with what came before the note so it is not credited for it. */
function calcSummary(rt) {
  const plural = (n, w) => `${n ?? 0} ${w}${n === 1 ? "" : "s"}`;
  const parts = [plural(rt.calc_notes, "note"), plural(rt.calc_calls, "call")];
  if (rt.calc_calls_before_note != null) {
    parts.push(`${rt.calc_calls_before_note} before the note, ${rt.calc_calls_after_note} after`);
    if (rt.calc_loaded_before_first_note) parts.push("already loaded before the note");
  } else if (rt.calc_note_followed != null) {
    parts.push(rt.calc_note_followed ? "note followed" : "note not followed"); // older traces
  }
  if (rt.delegation_prompt_names_calc) parts.push("the delegating prompt named it");
  if (rt.interpreter_calls_calc_could_run != null) {
    parts.push(`${rt.interpreter_calls_calc_could_run} of ${plural(rt.interpreter_calls, "interpreter call")} it could run`);
  }
  return parts.join(", ");
}

function renderTraceSummary(run) {
  const box = $("#trace-summary");
  const end = run.end || {};
  const d = end.decisions || {};
  const rt = end.routing || {};
  const concl = d.conclusion || {};
  const conf = concl.confidence;
  const head = el("div", { class: "verdict" },
    el("h3", { text: run.end ? (concl.recommended || "Finished; no recommendation found in the report") : "Run not finished (no SubagentStop yet)" }),
    conf ? el("span", { class: `pill ${CONF_TONE[conf] || "native"}`, text: `Confidence ${conf}` }) : null);
  const deleg = rt.delegation || null;
  const delegText = !deleg ? "–" : deleg.action === "suggest"
    ? `note sent (${vsThreshold(deleg.p, deleg.applied_threshold)})` : `no note (p ${deleg.p == null ? "–" : pct(deleg.p)})`;
  const gate = d.gate || {};
  const bands = Object.entries(gate.bands || {});
  const criteria = gate.criteria || {};
  const gt = d.ground_truths || {};
  const chains = d.chains || [];
  const confCount = {};
  chains.forEach((c) => { confCount[c.confidence || "?"] = (confCount[c.confidence || "?"] || 0) + 1; });
  const stats = [
    ["Duration", minutes(end.duration_s)],
    ["Sections written", end.sections_written ?? run.recs.filter((r) => r.kind === "section_written").length],
    ["Report restarts / revisions", `${end.report_restarts ?? 0} / ${end.report_revisions ?? 0}`],
    ["References opened", el("span", { class: "chips" }, ...(end.references_read || run.recs.filter((r) => r.kind === "reference_read").map((r) => r.file))
      .map((f) => chip(f.replace(/\.md$/, ""), "native")))],
    ["Delegation", delegText],
    ["Calculator", calcSummary(rt)],
    ["Self-Audit Gate", bands.length ? el("span", { class: "chips" }, ...bands.map(([n, b]) =>
      chip(`${n} ${b}`, BAND_TONE[b], criteria[n] ? `Criterion ${n}: ${criteria[n]}` : `Criterion ${n}`))) : "–"],
    ["Fix/Repeat", run.end ? fixRepeat(d) : "–"],
    ["Ground truths", gt.count == null ? "–" : `${gt.count}, ${(gt.unverified || []).length} not read at source`],
    ["Chains", chains.length ? el("span", { class: "chips" }, chip(String(chains.length), "native"),
      ...Object.entries(confCount).map(([c, n]) => chip(`${n} ${c}`, CONF_TONE[c]))) : "–"],
    ["Failed calls", end.tool_failures ?? run.recs.filter((r) => r.kind === "tool_failed").length],
    ["Source checks", end.sources ? `${end.sources.ok} ok, ${end.sources.reported_missing} without the content, ${end.sources.failed} failed${Object.keys(end.sources.failed_hosts || {}).length ? ` (${Object.keys(end.sources.failed_hosts).join(", ")})` : ""}` : "–"],
  ];
  const grid = el("div", { class: "stats" }, ...stats.map(([k, v]) => el("div", { class: "stat" },
    el("span", { class: "stat-k", text: k }), el("span", { class: "stat-v" }, v))));
  const foot = el("p", { class: "small-print" },
    `Session ${trace.data.session}, agent ${String(run.id).slice(0, 12)}. `,
    end.analysis ? el("code", { text: end.analysis }) : null);
  const links = trace.data.audit && trace.data.audit.length
    ? el("div", { class: "actions" }, el("button", { type: "button", class: "secondary",
      onclick: () => openReplay(trace.data.session), text: `Step through the router's ${trace.data.audit.length} checkpoints` }))
    : null;
  box.replaceChildren(head, grid, foot, links);
}

/** A shell call that runs an interpreter (``math`` in traces before 2026-09-30). */
const interp = (rec) => Boolean(rec.interpreter ?? rec.math);

function traceItem(rec, t0) {
  const dt = t0 == null ? 0 : (when(rec.ts) - t0) / 1000;
  const at = t0 == null || rec.kind === "run_end" ? "" : dt < 0 ? " (before the run)" : ` +${clock(dt)}`;
  const kind = (text, cls = "") => ({ kind: `${text}${at}`, cls });
  let meta; let body;
  switch (rec.kind) {
    case "run_start":
      meta = kind("Agent starts", "prompt");
      body = el("span", { text: `first-principles run ${String(rec.agent_id || "").slice(0, 12)}` });
      break;
    case "reference_read":
      meta = kind("Opens a reference");
      body = el("span", {}, el("span", { class: "tool-name", text: rec.file }), ` ${rec.meaning}`);
      break;
    case "section_written":
      meta = kind("Writes to the report", "write");
      body = el("div", {},
        el("ul", { class: "headings" }, ...(rec.headings || []).map((h) => el("li", { text: h }))),
        rec.restarts ? chip("empties the report first", "enforce") : null,
        rec.revises ? chip("also revises earlier text", "gated") : null,
        (rec.headings || []).length ? null : el("span", { class: "muted", text: `${rec.chars} characters, no heading` }));
      break;
    case "report_op":
      meta = kind(REPORT_OP[rec.op] || "Report file", rec.op === "revise" ? "write" : "");
      body = el("details", {}, el("summary", { text: rec.op === "revise" ? "rewrites part of the report" : "command" }), el("pre", { text: rec.command }));
      break;
    case "calc":
      meta = kind("Exact calculator", "calc");
      body = el("code", { text: `${rec.expression} = ${calcResult(rec.result)}` });
      break;
    case "shell":
      meta = kind(interp(rec) ? "Runs an interpreter" : "Shell command", interp(rec) ? "math" : "");
      body = el("details", {}, el("summary", { text: oneLine(rec.command) }), el("pre", { text: rec.command }));
      break;
    case "delegation":
      meta = kind("Main session delegates", "prompt");
      body = el("span", { text: `prompt of ${rec.prompt_chars} characters${rec.prompt_names_calc ? ", tells the agent to use the calculator" : ""}` });
      break;
    case "source":
      meta = kind(rec.tool === "WebSearch" ? "Searches for a source" : "Checks a source", rec.outcome === "ok" ? "" : "gated");
      body = el("span", {}, el("code", { text: rec.host || rec.target || "" }),
        rec.outcome === "reported_missing" ? chip("page did not have it", "gated") : null);
      break;
    case "tool_loaded":
      meta = kind("Loads tools", rec.calc ? "calc" : "");
      body = el("code", { text: rec.query });
      break;
    case "tool_failed":
      meta = kind(`${rec.tool_name ? rec.tool_name.replace(/^mcp__.*__/, "") : "Call"} failed`, "err");
      body = el("span", { text: rec.error || "" });
      break;
    case "run_end":
      meta = kind(`Agent stops after ${minutes(rec.duration_s)}`, "answer");
      body = el("details", {}, el("summary", { text: "its final message" }), el("pre", { text: rec.final_message || "" }));
      break;
    default:
      meta = kind(rec.kind);
      body = el("pre", { text: JSON.stringify(rec, null, 1) });
  }
  return el("li", { class: `tl ${meta.cls}` }, el("div", { class: "tl-kind", text: meta.kind }), el("div", { class: "tl-body" }, body));
}

function routerItem(rec, t0) {
  const at = t0 == null ? "" : ` +${clock((when(rec.ts) - t0) / 1000)}`;
  const name = rec.entry_id ? optionMeta(rec.entry_id).name : "an option";
  const p = rec.entry_id ? (rec.probabilities || {})[rec.entry_id] : null;
  return el("li", { class: `tl cp ${rec.action}` },
    el("div", { class: "tl-kind", text: `Router ${rec.action === "enforce" ? "blocks" : "note"}${at}` }),
    el("div", { class: "tl-body" }, `Points to ${name} on the next ${rec.tool_name || "call"}: `,
      el("b", { text: vsThreshold(p, appliedThreshold(rec)) || rec.reason })));
}

/** Consecutive calculator calls (or plain shell calls) fold into one item. */
function groupItem(recs, t0) {
  const first = recs[0];
  const at = t0 == null ? "" : ` +${clock((when(first.ts) - t0) / 1000)}`;
  const calc = first.kind === "calc";
  const label = calc ? `Exact calculator, ${recs.length} calls` : `${recs.length} shell commands`;
  const lines = recs.map((r) => (calc ? `${r.expression} = ${calcResult(r.result)}` : oneLine(r.command)));
  return el("li", { class: `tl ${calc ? "calc" : ""}` }, el("div", { class: "tl-kind", text: `${label}${at}` }),
    el("div", { class: "tl-body" }, el("details", {}, el("summary", { text: lines[0].slice(0, 90) + (recs.length > 1 ? ` … and ${recs.length - 1} more` : "") }),
      el("pre", { text: lines.join("\n") }))));
}

function renderTraceTimeline(run) {
  const t0 = run.start ? when(run.start.ts) : null;
  const foldable = (r) => r.kind === "calc" || (r.kind === "shell" && !interp(r));
  const items = [];
  for (const r of run.recs) {
    const last = items[items.length - 1];
    if (foldable(r) && last && last.group && last.group[0].kind === r.kind) last.group.push(r);
    else items.push({ ts: when(r.ts), order: 1, group: foldable(r) ? [r] : null, rec: r });
  }
  const all = [
    ...items.map((x) => ({ ...x, node: () => (x.group && x.group.length > 1 ? groupItem(x.group, t0) : traceItem(x.rec, t0)) })),
    ...routerMarks(run).map((r) => ({ ts: when(r.ts), order: 0, node: () => routerItem(r, t0) })),
  ].sort((a, b) => a.ts - b.ts || a.order - b.order);
  $("#trace-timeline").replaceChildren(...all.map((x) => x.node()));
}

function section(title, open, ...kids) {
  return el("details", { class: "decision", open }, el("summary", { text: title }), ...kids);
}

function renderDecisions(run) {
  const box = $("#trace-decisions");
  const d = (run.end || {}).decisions;
  if (!d) {
    box.replaceChildren(el("p", { class: "muted", text: run.end ? "The report file could not be read when the agent stopped." : "Decisions are read from the report when the agent stops." }));
    return;
  }
  const out = [];
  if (d.re_entry) out.push(el("p", { class: `re-entry ${d.re_entry.fired ? "fired" : ""}` },
    chip(d.re_entry.fired ? "re-entry fired" : "no re-entry", d.re_entry.fired ? "gated" : "native"), ` ${d.re_entry.disclosure}`));
  for (const c of d.contradictions || []) out.push(el("p", { class: "re-entry" }, chip("contradiction", "enforce"), ` ${c}`));
  if (d.run_mode) out.push(el("p", { class: "re-entry" }, chip(`mode ${d.run_mode}`, "outline"), ` ${d.run_mode_line || ""}`));
  const a = d.assumptions || { rows: [] };
  out.push(section(`Assumptions (${a.count || 0})`, true,
    el("p", { class: "chips" }, ...Object.entries(a.by_verdict || {}).map(([v, n]) => chip(`${n} ${v}`, VERDICT_TONE[v])),
      ...Object.entries(a.by_type || {}).map(([t, n]) => chip(`${n} ${t}`, "outline"))),
    el("table", { class: "dtable" },
      el("thead", {}, el("tr", {}, el("th", { text: "Assumption" }), el("th", { text: "Type" }), el("th", { text: "Verdict" }))),
      el("tbody", {}, ...(a.rows || []).map((r) => el("tr", {}, el("td", { text: r.assumption }), el("td", { text: r.type }),
        el("td", {}, chip(r.verdict || "?", VERDICT_TONE[r.verdict]))))))));
  const gt = d.ground_truths || {};
  out.push(section(`Ground truths (${gt.count || 0}, ${(gt.unverified || []).length} not read at source)`, false,
    (gt.unverified || []).length
      ? el("p", { class: "chips" }, ...gt.unverified.map((g) => chip(g, "gated", "reported or unverified, not read at source")))
      : el("p", { class: "muted", text: "Every ground truth was read at source." })));
  out.push(section(`Derivation chains (${(d.chains || []).length})`, true,
    el("ul", { class: "dlist" }, ...(d.chains || []).map((c) => el("li", {},
      chip(c.confidence || "?", CONF_TONE[c.confidence]), " ", el("b", { text: c.id }), c.tag ? ` ${c.tag}` : "", `: ${c.title}`)))));
  const gate = d.gate || {};
  out.push(section(`Self-Audit Gate (${gate.passes || 0} scoring pass${gate.passes === 1 ? "" : "es"})`, true,
    el("table", { class: "dtable" }, el("tbody", {}, ...Object.entries(gate.bands || {}).map(([n, b]) => el("tr", {},
      el("td", { text: `${n}. ${(gate.criteria || {})[n] || "Criterion"}` }), el("td", {}, chip(b, BAND_TONE[b])))))),
    gate.result ? el("p", { class: "small-print", text: gate.result }) : null));
  out.push(section(`Dead ends (${(d.dead_ends || []).length})`, false,
    el("ul", { class: "dlist" }, ...(d.dead_ends || []).map((x) => el("li", { text: x })))));
  out.push(section(`Techniques not applied (${(d.techniques_not_applied || []).length})`, false,
    el("ul", { class: "dlist" }, ...(d.techniques_not_applied || []).map((t) => el("li", {},
      el("b", { text: t.technique }), ` (Phase ${t.phase}): ${t.reason}`)))));
  out.push(section(`Report sections (${(d.sections || []).length})`, false,
    el("ol", { class: "dlist" }, ...(d.sections || []).map((h) => el("li", { text: h })))));
  box.replaceChildren(...out);
}

function initTrace() {
  $("#trace-select").addEventListener("change", (e) => loadTrace(e.target.value));
  $("#trace-refresh").addEventListener("click", () => loadTraceSessions($("#trace-select").value));
}

/* -- tabs & boot ---------------------------------------------------------------- */

const TABS = { play: "Playground", live: "Live agent", replay: "Replay", trace: "Trace" };

function selectTab(name) {
  for (const key of Object.keys(TABS)) {
    const on = key === name;
    $(`#tab-${key}`).setAttribute("aria-selected", on ? "true" : "false");
    $(`#view-${key}`).hidden = !on;
  }
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (name === "replay" && !replay.records.length) loadSessions();
  if (name === "trace" && !trace.data) loadTraceSessions();
}

async function boot() {
  for (const key of Object.keys(TABS)) $(`#tab-${key}`).addEventListener("click", () => selectTab(key));
  const [catalog, backends] = await Promise.all([api("/api/catalog"), api("/api/backends")]);
  state.catalog = catalog;
  state.threshold = catalog.threshold;
  catalog.entries.forEach((e) => state.entries.set(e.id, e));
  state.backends = backends.backends;
  $("#shell-note").textContent = catalog.live && catalog.live.allow_shell
    ? "Shell and web tools (Bash, WebFetch) are enabled: the server was started with allow_shell."
    : "Shell and web tools (Bash, WebFetch) are disabled. The router still sees those calls first; the SDK then refuses them. Start the server with --allow-shell to enable them.";
  showThreshold(catalog.threshold);
  if (catalog.mode === "enforce") $("input[name=mode][value=enforce]").checked = true;
  for (const sel of [$("#backend"), $("#live-backend")]) {
    sel.replaceChildren(...state.backends.map((b) => el("option", { value: b.name, disabled: !b.available },
      b.available ? b.name : `${b.name} (not available)`)));
    sel.value = backends.default;
  }
  initPlayground();
  initLive();
  initReplay();
  initTrace();
  const hash = location.hash.slice(1);
  selectTab(TABS[hash] ? hash : "play");
}

boot().catch((err) => {
  $("#board").replaceChildren(el("p", { class: "board-empty", text: `Could not reach the demo server: ${err.message}` }));
});
