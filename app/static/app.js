"use strict";
/* No innerHTML / outerHTML / insertAdjacentHTML / document.write anywhere in this
 * file. Every piece of text that can contain model/user-generated content
 * (task text, plan text, gap text, candidate code, failure text) is attached
 * with .textContent only, so it is never parsed as markup -- a "<script>" in
 * a candidate's code is just characters in a <pre>, not executable HTML. */

const POLL_MS = 600;

function qs(id) {
  return document.getElementById(id);
}

function el(tag, opts, children) {
  const node = document.createElement(tag);
  opts = opts || {};
  if (opts.className) node.className = opts.className;
  if (opts.text !== undefined) node.textContent = opts.text;
  if (opts.title !== undefined) node.title = opts.title;
  if (opts.attrs) {
    for (const k in opts.attrs) node.setAttribute(k, opts.attrs[k]);
  }
  if (opts.onclick) node.addEventListener("click", opts.onclick);
  (children || []).forEach((c) => node.appendChild(c));
  return node;
}

async function getJSON(url) {
  const resp = await fetch(url);
  return { status: resp.status, body: await resp.json() };
}

async function postJSON(url, obj) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(obj),
  });
  return { status: resp.status, body: await resp.json() };
}

const EXAMPLES = [
  "Move the arm's gripper to the target point.",
  "Pick up the cube and place it in the bin.",
  "Stack two cubes on top of each other.",
];

let pollTimer = null;
let replayMode = false;

function setMode(mode) {
  const pill = qs("mode-pill");
  pill.textContent = mode + " mode";
  pill.className = "mode-pill " + mode;
  replayMode = mode === "replay";
  qs("task-input").disabled = replayMode;
  qs("run-btn").disabled = replayMode;
  qs("run-hint").textContent = replayMode
    ? "Serving a recorded run (read-only)."
    : "";
}

function renderExamples() {
  const box = qs("examples");
  box.replaceChildren();
  EXAMPLES.forEach((task) => {
    box.appendChild(
      el("button", {
        className: "example-btn",
        text: task,
        attrs: { type: "button" },
        onclick: () => {
          qs("task-input").value = task;
        },
      })
    );
  });
}

function showOnly(name) {
  ["empty-state", "loading-state", "error-state", "result-state"].forEach((id) => {
    qs(id).classList.toggle("hidden", id !== name);
  });
}

function fmtSeconds(s) {
  return s === null || s === undefined ? "—" : s.toFixed(1);
}
function fmtCost(c) {
  return c === null || c === undefined ? "—" : "$" + c.toFixed(4);
}

function renderCounters(result) {
  qs("c-seconds").textContent = fmtSeconds(result.elapsed_s);
  qs("c-tokens").textContent = result.tokens ?? "—";
  qs("c-cost").textContent = fmtCost(result.cost_usd);
  qs("c-simruns").textContent = result.sim_runs ?? "—";
  qs("c-library").textContent = result.library_size ?? "—";
}

function statusLabel(status) {
  if (status === "verified") return "Verified";
  if (status === "failed_verification") return "Failed verification";
  if (status === "rejected_preflight") return "Rejected by pre-flight";
  return status;
}

function renderSeedDots(score) {
  const wrap = el("div", { className: "seed-dots" });
  if (!score) return wrap;
  score.seeds.forEach((s) => {
    const dot = el("button", {
      className: "seed-dot " + s.status,
      attrs: { type: "button", "aria-label": `seed ${s.seed}: ${s.status}` },
      title: `seed ${s.seed}: ${s.status}\n${s.detail}`,
    });
    wrap.appendChild(dot);
  });
  return wrap;
}

function renderCandidateCard(group, winner) {
  const latest = group.attempts[group.attempts.length - 1];
  const isWinner = winner && winner.idx === group.idx;
  const card = el("div", {
    className: `candidate-card status-${latest.status}` + (isWinner ? " is-winner" : ""),
  });
  const head = el("div", { className: "candidate-head" }, [
    el("span", { className: "candidate-title", text: `Candidate ${group.idx + 1}` }),
    el("span", { className: "pill status-" + latest.status, text: statusLabel(latest.status) }),
  ]);
  card.appendChild(head);

  if (group.attempts.length > 1) {
    card.appendChild(
      el("div", {
        className: "seed-detail",
        text: `${group.attempts.length} attempts (repaired ${group.attempts.length - 1}x)`,
      })
    );
  }

  if (!latest.preflight_ok) {
    card.appendChild(
      el("div", {
        className: "preflight-reason",
        text: latest.preflight_problems.join("; "),
      })
    );
  } else if (latest.score) {
    card.appendChild(renderSeedDots(latest.score));
    card.appendChild(
      el("div", {
        className: "seed-detail",
        text: `success rate ${(latest.score.success_rate * 100).toFixed(0)}% · ${latest.score.mean_seconds.toFixed(1)}s/run`,
      })
    );
    if (!latest.score.ok && latest.score.failure_text) {
      const details = el("details", {}, [
        el("summary", { text: "failure detail" }),
        el("div", { className: "seed-detail", text: latest.score.failure_text }),
      ]);
      card.appendChild(details);
    }
  }

  const codeDetails = el("details", {}, [
    el("summary", { text: "view script" }),
    el(
      "div",
      { className: "code-viewer" },
      [el("pre", { text: latest.script || "(empty)" })]
    ),
  ]);
  card.appendChild(codeDetails);

  return card;
}

function renderCandidates(result) {
  const box = qs("candidates");
  box.replaceChildren();
  result.candidates.forEach((group) => {
    box.appendChild(renderCandidateCard(group, result.winner));
  });
}

function renderTimeline(result) {
  const box = qs("timeline");
  box.replaceChildren();
  if (!result.repair_rounds.length) {
    box.appendChild(el("div", { className: "timeline-empty", text: "No repair rounds were needed." }));
    return;
  }
  result.repair_rounds.forEach((r, i) => {
    box.appendChild(
      el("div", { className: "timeline-row" }, [
        el("span", { text: `Round ${i + 1} — candidate ${r.candidate_idx + 1}` }),
        el("span", { className: "pill status-" + r.before_status, text: statusLabel(r.before_status) }),
        el("span", { className: "timeline-arrow", text: "→" }),
        el("span", { className: "pill status-" + r.after_status, text: statusLabel(r.after_status) }),
      ])
    );
  });
}

function renderMediaSlot(attempt) {
  const wrap = el("div", { className: "media-slot" });
  const clipPath = attempt && attempt.clip_path;
  if (!clipPath) {
    wrap.appendChild(el("p", { className: "hint", text: "no recording" }));
    return wrap;
  }
  const ext = (clipPath.split(".").pop() || "").toLowerCase();
  const src = "/api/media?path=" + encodeURIComponent(clipPath);
  if (ext === "mp4" || ext === "webm") {
    const video = document.createElement("video");
    video.src = src;
    video.controls = true;
    video.muted = true;
    video.playsInline = true;
    video.className = "media-player";
    wrap.appendChild(video);
  } else if (ext === "gif" || ext === "png") {
    const img = document.createElement("img");
    img.src = src;
    img.alt = "recording of the winning run";
    img.className = "media-player";
    wrap.appendChild(img);
  } else {
    wrap.appendChild(el("p", { className: "hint", text: "no recording" }));
  }
  return wrap;
}

// The green "winner-panel" class and the literal word "Winner" are reserved
// for a VERIFIED run. Anything else -- PARTIAL (a winner candidate exists but
// didn't pass every seed) or FAILED (no candidate did) -- renders as "Best
// attempt (not verified)" in a neutral/red style with the failure reason
// shown, never the verified styling or wording.
function renderWinner(result) {
  const box = qs("winner");
  const titleNode = qs("winner-title");
  box.replaceChildren();
  const verified = result.status === "VERIFIED";
  titleNode.textContent = verified ? "Winner" : "Best attempt (not verified)";

  if (!result.winner) {
    box.className = "winner-panel none";
    box.appendChild(
      el("p", {
        text:
          "No verified script. Closest template + gap: " +
          result.template.name +
          (result.gap ? " — " + result.gap : " (exact match; no gap)") +
          ". This run did not succeed; nothing here should be used as-is.",
      })
    );
    return;
  }
  const group = result.candidates.find((g) => g.idx === result.winner.idx);
  const attempt = group.attempts[result.winner.attempt];

  box.className = verified ? "winner-panel" : "winner-panel not-verified";

  const copyBtn = el("button", {
    className: "btn btn-copy",
    text: "Copy code",
    attrs: { type: "button", "aria-live": "polite" },
  });
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(attempt.script);
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "Copy code"), 1200);
    } catch (e) {
      copyBtn.textContent = "Copy failed";
    }
  });

  const headLabel = verified
    ? `Candidate ${group.idx + 1} (${attempt.stage})`
    : `Candidate ${group.idx + 1} (${attempt.stage}) — not verified`;
  box.appendChild(
    el("div", { className: "winner-head" }, [el("span", { text: headLabel }), copyBtn])
  );

  if (!verified) {
    const reason =
      (attempt.score && attempt.score.failure_text) ||
      "No candidate passed every seed; this is the closest attempt, shown for reference only, and must not be used as-is.";
    box.appendChild(el("p", { className: "winner-reason", text: "Why this is not verified: " + reason }));
  }

  box.appendChild(el("div", { className: "code-viewer" }, [el("pre", { text: attempt.script })]));
  box.appendChild(el("h3", { className: "media-label", text: "Recording" }));
  box.appendChild(renderMediaSlot(attempt));
}

// A result with no recognized source is treated as the least trustworthy
// case (not silently rendered as if it were mock or live), so this always
// writes *some* unmissable badge text -- there is no path through
// renderResult() that leaves source-badge empty.
function renderSourceBadge(result) {
  const box = qs("source-badge");
  box.replaceChildren();
  const source = result.source;
  box.className = "source-badge source-" + (source || "unknown");
  let text;
  if (source === "mock") {
    text = "MOCK: scripted LLM, no model was called";
  } else if (source === "recorded_live") {
    text =
      "Recorded live run: " +
      (result.model || "unknown model") +
      ", " +
      (result.date || "unknown date") +
      ", prompt " +
      (result.prompt_version || "unknown");
  } else if (source === "live") {
    text = "LIVE run: " + (result.model || "unknown model") + ", real model call, real simulation";
  } else {
    text = "UNKNOWN SOURCE — this run was not labeled mock/live/recorded_live; treat with caution";
  }
  box.textContent = text;
}

function renderTemplateBadge(result) {
  const box = qs("template-badge");
  box.replaceChildren();
  box.className = "template-badge" + (result.exact_match ? " exact" : "");
  box.appendChild(el("span", { className: "tpl-name", text: result.template.name }));
  box.appendChild(el("span", { text: result.template.description }));
  if (!result.exact_match) {
    box.appendChild(el("span", { className: "tpl-gap", text: "Gap: " + result.gap }));
  }
}

function renderOutcomeBanner(result) {
  const box = qs("outcome-banner");
  box.className = "outcome-banner " + result.status;
  const text =
    result.status === "VERIFIED"
      ? "Verified"
      : result.status === "PARTIAL"
      ? "Partial — no candidate met the bar"
      : "Failed — no verified script";
  box.textContent = text;
}

// result.tavily is only ever a real, sourced research result (see
// tavily_failure_modes_for_result() in app/server.py) or null -- null covers
// Tavily being off for this run (mock/replay, or a live run with SIMSMITH_TAVILY
// unset), never a mocked stand-in rendered as if it were real.
function renderTavilyPanel(result) {
  const box = qs("tavily-panel");
  box.replaceChildren();
  const tavily = result.tavily;
  if (!tavily) {
    const label =
      result.source === "mock"
        ? "MOCK run: no real Tavily research was performed."
        : "Tavily research was off for this run (set SIMSMITH_TAVILY=1 and TAVILY_API_KEY to enable).";
    box.appendChild(el("p", { className: "hint tavily-off", text: label }));
    return;
  }
  if (tavily.items.length === 0) {
    box.appendChild(el("p", { className: "hint", text: "Tavily ran but returned no usable failure modes." }));
    return;
  }
  const list = el("ul", { className: "tavily-list" });
  tavily.items.forEach((item) => {
    const link = el("a", { text: item.source_title || item.source_url });
    link.setAttribute("href", item.source_url);
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
    const li = el("li", { className: "tavily-item" }, [
      el("div", { className: "tavily-mode", text: item.mode }),
      el("div", { className: "tavily-why", text: item.why_it_matters }),
      el("div", { className: "tavily-source" }, [el("span", { text: "source: " }), link]),
    ]);
    list.appendChild(li);
  });
  box.appendChild(list);
}

function renderResult(result) {
  renderSourceBadge(result);
  renderCounters(result);
  renderTemplateBadge(result);
  renderOutcomeBanner(result);
  qs("notes").textContent = result.notes.join(" ");
  renderTavilyPanel(result);
  renderCandidates(result);
  renderTimeline(result);
  renderWinner(result);
  renderStageStrip(result);
  showOnly("result-state");
}

// ---- pipeline stage strip (static row that lights up from the finished
// run's own data -- there is no live progress hook yet, so nothing here
// updates mid-run). ----------------------------------------------------------

const STAGES = ["intake", "generate", "preflight", "execute", "verify", "repair"];

function stageNode(stage) {
  return document.querySelector(`.stage-item[data-stage="${stage}"]`);
}

function clearStages() {
  STAGES.forEach((s) => {
    const node = stageNode(s);
    if (node) node.classList.remove("active");
  });
}

function computeActiveStages(result) {
  const attempts = result.candidates.flatMap((g) => g.attempts);
  return {
    intake: true,
    generate: attempts.length > 0,
    preflight: attempts.some((a) => a.preflight_ok !== undefined),
    execute: (result.sim_runs || 0) > 0,
    verify: attempts.some((a) => !!a.score),
    repair: (result.repair_rounds || []).length > 0,
  };
}

function renderStageStrip(result) {
  const active = computeActiveStages(result);
  STAGES.forEach((s) => {
    const node = stageNode(s);
    if (node) node.classList.toggle("active", !!active[s]);
  });
}

// ---- demo / presentation mode (?present=1 or the header toggle) -----------

function applyPresentMode(on) {
  document.body.classList.toggle("present-mode", on);
  const btn = qs("present-toggle");
  btn.textContent = on ? "Exit present mode" : "Present mode";
  btn.setAttribute("aria-pressed", on ? "true" : "false");
}

function initPresentMode() {
  const params = new URLSearchParams(window.location.search);
  let on = params.get("present") === "1";
  applyPresentMode(on);
  qs("present-toggle").addEventListener("click", () => {
    on = !on;
    applyPresentMode(on);
  });
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollStatus() {
  const { body } = await getJSON("/api/status");
  if (body.state === "running") {
    showOnly("loading-state");
    return;
  }
  stopPolling();
  qs("run-btn").disabled = replayMode;
  if (body.state === "error") {
    qs("error-state").textContent = "Run failed: " + body.error;
    showOnly("error-state");
    return;
  }
  if (body.state === "done" && body.result) {
    renderResult(body.result);
    return;
  }
  showOnly("empty-state");
}

async function onRunClick() {
  const task = qs("task-input").value.trim();
  if (!task) return;
  qs("run-btn").disabled = true;
  clearStages();
  showOnly("loading-state");
  const { status, body } = await postJSON("/api/run", { task: task });
  if (status === 409) {
    qs("error-state").textContent = body.error;
    showOnly("error-state");
    qs("run-btn").disabled = replayMode;
    return;
  }
  if (status >= 400) {
    qs("error-state").textContent = body.error || "request rejected";
    showOnly("error-state");
    qs("run-btn").disabled = replayMode;
    return;
  }
  stopPolling();
  pollTimer = setInterval(pollStatus, POLL_MS);
  pollStatus();
}

function fmtPct(v) {
  return v === null || v === undefined ? "—" : (v * 100).toFixed(0) + "%";
}

function renderBackendRow(b) {
  const tr = el("tr", {});
  tr.appendChild(el("td", { text: b.backend }));
  tr.appendChild(el("td", { text: b.seconds_per_run == null ? "—" : b.seconds_per_run.toFixed(2) }));
  tr.appendChild(el("td", { text: fmtPct(b.pass_rate_handwritten) }));
  tr.appendChild(el("td", { text: (b.tasks_covered || []).join(", ") || "—" }));
  tr.appendChild(el("td", { text: b.setup_notes || "" }));
  tr.appendChild(el("td", { text: b.measured ? "measured" : "not measured" }));
  return tr;
}

async function loadBackendComparison() {
  try {
    const { body } = await getJSON("/api/backend-comparison");
    qs("backend-fidelity-note").textContent = body.note || "";
    const tbody = qs("backend-tbody");
    (body.backends || []).forEach((b) => tbody.appendChild(renderBackendRow(b)));
  } catch (e) {
    qs("backend-fidelity-note").textContent = "Backend comparison data unavailable.";
  }
}

async function init() {
  renderExamples();
  loadBackendComparison();
  initPresentMode();
  qs("run-btn").addEventListener("click", onRunClick);

  const { body } = await getJSON("/api/status");
  setMode(body.mode);
  if (body.library_size !== null && body.library_size !== undefined) {
    qs("c-library").textContent = body.library_size;
  }
  if (body.state === "done" && body.result) {
    renderResult(body.result);
  } else if (body.state === "running") {
    showOnly("loading-state");
    pollTimer = setInterval(pollStatus, POLL_MS);
  } else if (body.state === "error") {
    qs("error-state").textContent = "Run failed: " + body.error;
    showOnly("error-state");
  } else {
    showOnly("empty-state");
  }
}

init();
