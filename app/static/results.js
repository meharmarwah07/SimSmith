"use strict";
/* Same rule as app.js: every piece of text that can come from a data file
 * (model names, file names, setup notes, failure-category labels) is
 * attached with .textContent only, never innerHTML/outerHTML/document.write.
 * The conditions chart is plain SVG built with document.createElementNS --
 * the SVG-namespaced sibling of createElement -- for the same reason. */

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

function fmtSeconds(v) {
  return v === null || v === undefined ? "—" : v.toFixed(1) + "s";
}

function fmtCost(v) {
  return v === null || v === undefined ? "—" : "$" + v.toFixed(4);
}

function fmtPct(v) {
  return v === null || v === undefined ? "—" : (v * 100).toFixed(0) + "%";
}

// Canonical condition vocabulary (kept in sync by hand with
// simsmith/run_meta.py's VALID_CONDITIONS and server.py's detect_condition),
// ordered from least to most machinery -- this is also the bar order in the
// Conditions chart and the sort order for the Condition column in the
// table. "unknown" (a file whose condition couldn't be determined from any
// explicit field, n_candidates, or filename) always sorts last so it never
// hides ahead of a condition we're actually sure about.
const CONDITION_ORDER = [
  "single_shot", "best_of_5", "best_of_5_repair",
  "best_of_5_repair_library", "best_of_5_repair_skills",
];

const CONDITION_DISPLAY_NAMES = {
  single_shot: "Single-shot",
  best_of_5: "Best-of-5",
  best_of_5_repair: "Best-of-5 + repair",
  best_of_5_repair_library: "Best-of-5 + repair + library",
  best_of_5_repair_skills: "Best-of-5 + repair + skills",
};

// Tavily is an orthogonal flag server.py may set on a row (detect_tavily_on),
// not a condition of its own -- shown as a suffix, never folded into the
// condition identity used for grouping/sorting/coloring.
function conditionLabel(condition) {
  return CONDITION_DISPLAY_NAMES[condition] || condition;
}

function rowConditionLabel(row) {
  return conditionLabel(row.condition) + (row.tavily_on ? " (Tavily)" : "");
}

function conditionRank(c) {
  const i = CONDITION_ORDER.indexOf(c);
  return i === -1 ? CONDITION_ORDER.length : i;
}

function renderFailureBar(breakdown) {
  const wrap = el("div", {});
  const entries = Object.entries(breakdown || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    wrap.appendChild(el("span", { className: "hint", text: "no failures recorded" }));
    return wrap;
  }
  const total = entries.reduce((s, [, v]) => s + v, 0);
  const bar = el("div", { className: "fail-bar" });
  entries.forEach(([k, v]) => {
    const seg = el("div", {
      className: "fail-seg fail-" + k.replace(/[^a-z0-9]/gi, "_"),
      attrs: { title: `${k}: ${v}` },
    });
    seg.style.width = ((100 * v) / total).toFixed(1) + "%";
    seg.appendChild(el("span", { className: "sr-only", text: `${k}: ${v}` }));
    bar.appendChild(seg);
  });
  wrap.appendChild(bar);
  wrap.appendChild(
    el("div", { className: "fail-legend", text: entries.map(([k, v]) => `${k}: ${v}`).join(" · ") })
  );
  return wrap;
}

/* ---- classifier-version warning banner (one line, with the count of files
 * it affects) + a small per-row icon -- replaces the old full-text chip on
 * every row, which crowded the Failure breakdown column. ---- */

function renderClassifierBanner(rows) {
  const banner = qs("classifier-banner");
  const byFile = new Map();
  rows.forEach((r) => {
    if (!byFile.has(r.file)) byFile.set(r.file, false);
    if (r.low_confidence_labels) byFile.set(r.file, true);
  });
  const total = byFile.size;
  const affected = Array.from(byFile.values()).filter(Boolean).length;
  if (affected === 0) {
    banner.classList.add("hidden");
    banner.textContent = "";
    return;
  }
  banner.textContent =
    `${affected} of ${total} result file${total === 1 ? "" : "s"} ` +
    "use labels from an older failure classifier (classifier_version below 2, or missing); " +
    "hallucinated_api counts in those files may be over-counted. Pass rates are unaffected.";
  banner.classList.remove("hidden");
}

/* ---- grouped, sorted results table: model -> template -> prompt_version ->
 * condition -> rows, newest prompt version shown by default, newest row
 * first within each condition. ---- */

function groupRows(rows) {
  const byModel = new Map();
  rows.forEach((r) => {
    if (!byModel.has(r.model)) byModel.set(r.model, new Map());
    const byTemplate = byModel.get(r.model);
    if (!byTemplate.has(r.template)) byTemplate.set(r.template, new Map());
    const byPrompt = byTemplate.get(r.template);
    if (!byPrompt.has(r.prompt_version)) byPrompt.set(r.prompt_version, new Map());
    const byCondition = byPrompt.get(r.prompt_version);
    if (!byCondition.has(r.condition)) byCondition.set(r.condition, []);
    byCondition.get(r.condition).push(r);
  });
  return byModel;
}

// The "newest" prompt version for a model+template is the one whose rows
// include the most recent date -- prompt_version strings aren't reliably
// orderable (mujoco-v2 vs mujoco-v10), but the ISO date on each run is.
function newestPromptVersion(byPrompt) {
  let best = null;
  let bestDate = "";
  for (const [pv, byCondition] of byPrompt) {
    for (const rowsArr of byCondition.values()) {
      for (const r of rowsArr) {
        if (r.date && r.date > bestDate) {
          bestDate = r.date;
          best = pv;
        }
      }
    }
  }
  return best;
}

const RESULTS_COLSPAN = "8";

function groupHeaderRow(text, cls) {
  const tr = el("tr", { className: "group-row " + cls });
  tr.appendChild(el("td", { attrs: { colspan: RESULTS_COLSPAN }, text }));
  return tr;
}

function hintRow(text) {
  const tr = el("tr", { className: "group-row group-hint" });
  tr.appendChild(el("td", { attrs: { colspan: RESULTS_COLSPAN }, className: "hint", text }));
  return tr;
}

function renderRow(row) {
  const tr = el("tr", {});
  tr.appendChild(el("td", { text: rowConditionLabel(row) }));
  tr.appendChild(el("td", { text: row.date || "—" }));
  const ci =
    row.pass_rate === null || row.pass_rate === undefined
      ? "—"
      : `${fmtPct(row.pass_rate)} [${fmtPct(row.wilson_lo)}–${fmtPct(row.wilson_hi)}]`;
  tr.appendChild(el("td", { className: "ci-cell", text: ci }));
  tr.appendChild(el("td", { text: String(row.n) }));
  tr.appendChild(el("td", { text: fmtSeconds(row.mean_seconds) }));
  tr.appendChild(el("td", { text: fmtCost(row.mean_cost_usd) }));
  const failTd = el("td", { className: "fail-cell" });
  failTd.appendChild(renderFailureBar(row.failure_breakdown));
  if (row.low_confidence_labels) {
    failTd.appendChild(
      el("span", {
        className: "warn-icon",
        text: "⚠",
        attrs: { role: "img", "aria-label": "classifier warning" },
        title: "Labels from an older failure classifier may be wrong (hallucinated_api over-counted).",
      })
    );
  }
  tr.appendChild(failTd);
  tr.appendChild(el("td", { className: "file-cell", text: row.file || "" }));
  return tr;
}

let showOlderRuns = false;

function renderResultsTable(rows) {
  const tbody = qs("results-tbody");
  tbody.replaceChildren();
  const byModel = groupRows(rows);
  const models = Array.from(byModel.keys()).sort();

  models.forEach((model) => {
    tbody.appendChild(groupHeaderRow(`Model: ${model}`, "group-model"));
    const byTemplate = byModel.get(model);
    const templates = Array.from(byTemplate.keys()).sort();

    templates.forEach((template) => {
      tbody.appendChild(groupHeaderRow(`Template: ${template}`, "group-template"));
      const byPrompt = byTemplate.get(template);
      const newestPV = newestPromptVersion(byPrompt);
      const promptVersions = Array.from(byPrompt.keys()).sort((a, b) => {
        if (a === newestPV) return -1;
        if (b === newestPV) return 1;
        return a < b ? 1 : a > b ? -1 : 0;
      });

      let hiddenCount = 0;
      promptVersions.forEach((pv) => {
        const isNewest = pv === newestPV;
        if (!isNewest && !showOlderRuns) {
          hiddenCount += 1;
          return;
        }
        tbody.appendChild(
          groupHeaderRow(`Prompt ${pv}` + (isNewest ? "" : " (older)"), "group-prompt")
        );
        const byCondition = byPrompt.get(pv);
        const conditions = Array.from(byCondition.keys()).sort(
          (a, b) => conditionRank(a) - conditionRank(b)
        );
        conditions.forEach((condition) => {
          const condRows = byCondition
            .get(condition)
            .slice()
            .sort((a, b) => ((a.date || "") < (b.date || "") ? 1 : (a.date || "") > (b.date || "") ? -1 : 0));
          condRows.forEach((r) => tbody.appendChild(renderRow(r)));
        });
      });

      if (hiddenCount > 0) {
        tbody.appendChild(
          hintRow(
            `${hiddenCount} older prompt-version run${hiddenCount > 1 ? "s" : ""} hidden — ` +
              'check "Show older runs" above to see them.'
          )
        );
      }
    });
  });
}

function wireOlderRunsToggle(rows) {
  const toggle = qs("show-older-toggle");
  toggle.addEventListener("change", () => {
    showOlderRuns = toggle.checked;
    renderResultsTable(rows);
  });
}

/* ---- Conditions chart: one block per (model, template), scoped to a
 * single prompt version at a time -- a block never mixes prompt versions
 * into one bar. Each block defaults to the newest version that has any
 * data and offers a picker to switch to any other single version, or to
 * "All versions" (the only mode that shows more than one version in the
 * same block, and the only one that shows the not-comparable warning).
 * A condition with more than one run for the chosen version never gets
 * pooled into one averaged bar -- every run gets its own marker+interval
 * on the row and its own line in the list underneath. ---- */

const SVG_NS = "http://www.w3.org/2000/svg";
const ALL_VERSIONS = "__all__";

function svgEl(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const k in attrs || {}) node.setAttribute(k, attrs[k]);
  if (text !== undefined) node.textContent = text;
  return node;
}

function groupByModelTemplate(rows) {
  const map = new Map();
  rows.forEach((r) => {
    const key = r.model + "\u0000" + r.template;
    if (!map.has(key)) map.set(key, { model: r.model, template: r.template, rows: [] });
    map.get(key).rows.push(r);
  });
  return Array.from(map.values()).sort((a, b) => {
    if (a.model !== b.model) return a.model < b.model ? -1 : 1;
    return a.template < b.template ? -1 : a.template > b.template ? 1 : 0;
  });
}

// The default prompt version for a block is the one whose rows include the
// most recent date -- prompt_version strings aren't reliably orderable
// (mujoco-v2 vs mujoco-v10), but each run's ISO date is.
function pickDefaultPromptVersion(rows) {
  let best = null;
  let bestDate = "";
  rows.forEach((r) => {
    if (r.date && r.date > bestDate) {
      bestDate = r.date;
      best = r.prompt_version;
    }
  });
  return best;
}

// One row per condition. Every run for that condition (there may be more
// than one file/date on record) gets its own marker+whisker on the row and
// its own line in the list below -- this never averages/pools multiple
// runs into a single bar.
function renderConditionRow(condition, runs, showVersionPerRun) {
  const label = conditionLabel(condition);
  const wrap = el("div", { className: "condition-row" });
  wrap.appendChild(el("div", { className: "condition-row-label", text: label }));

  const width = 560;
  const height = 40;
  const trackHeight = 16;
  const trackY = (height - trackHeight) / 2;
  const midY = height / 2;

  const svg = svgEl("svg", {
    width: String(width), height: String(height), viewBox: `0 0 ${width} ${height}`,
    role: "img", "aria-label": `${label}: ${runs.length} run${runs.length === 1 ? "" : "s"}`,
    class: "condition-row-svg",
  });
  svg.appendChild(
    svgEl("rect", {
      x: "0", y: String(trackY), width: String(width), height: String(trackHeight), class: "chart-track",
    })
  );

  runs.forEach((r) => {
    const pr = Math.max(0, Math.min(1, r.pass_rate));
    const x = pr * width;
    const loX = Math.max(0, Math.min(1, r.wilson_lo)) * width;
    const hiX = Math.max(0, Math.min(1, r.wilson_hi)) * width;
    svg.appendChild(
      svgEl("line", { x1: String(loX), x2: String(hiX), y1: String(midY), y2: String(midY), class: "chart-whisker" })
    );
    [loX, hiX].forEach((lx) => {
      svg.appendChild(
        svgEl("line", {
          x1: String(lx), x2: String(lx), y1: String(trackY - 5), y2: String(trackY + trackHeight + 5),
          class: "chart-whisker",
        })
      );
    });
    svg.appendChild(svgEl("circle", { cx: String(x), cy: String(midY), r: "7", class: "chart-marker" }));
  });
  wrap.appendChild(svg);

  // Always listed below the row -- for a single run this is just one line;
  // for several, every run's own n/interval/date/file is spelled out rather
  // than collapsed into one number.
  const list = el("ul", { className: "run-list" });
  runs.forEach((r) => {
    const parts = [`n=${r.n}`, `${fmtPct(r.pass_rate)} [${fmtPct(r.wilson_lo)}–${fmtPct(r.wilson_hi)}]`];
    if (showVersionPerRun) parts.push(`prompt ${r.prompt_version}`);
    if (r.tavily_on) parts.push("tavily");
    parts.push(r.date || "unknown date");
    if (r.file) parts.push(r.file);
    list.appendChild(el("li", { text: parts.join(" · ") }));
  });
  wrap.appendChild(list);

  return wrap;
}

// A same-shaped row for a canonical condition with zero runs for this
// model/template/prompt-version -- a dashed, marker-less track and a "Not
// run" label, so a condition nobody has tried yet reads as genuinely
// untested rather than being indistinguishable from "scrolled off the page"
// or silently omitted.
function renderNotRunConditionRow(condition) {
  const label = conditionLabel(condition);
  const wrap = el("div", { className: "condition-row condition-row-not-run" });
  wrap.appendChild(el("div", { className: "condition-row-label", text: label }));

  const width = 560;
  const height = 40;
  const trackHeight = 16;
  const trackY = (height - trackHeight) / 2;

  const svg = svgEl("svg", {
    width: String(width), height: String(height), viewBox: `0 0 ${width} ${height}`,
    role: "img", "aria-label": `${label}: not run`,
    class: "condition-row-svg condition-row-svg-not-run",
  });
  svg.appendChild(
    svgEl("rect", {
      x: "0", y: String(trackY), width: String(width), height: String(trackHeight),
      class: "chart-track chart-track-empty",
    })
  );
  svg.appendChild(
    svgEl("text", {
      x: String(width / 2), y: String(height / 2 + 4), class: "chart-not-run-label", "text-anchor": "middle",
    }, "Not run")
  );
  wrap.appendChild(svg);

  const list = el("ul", { className: "run-list" });
  list.appendChild(el("li", { className: "hint", text: "No data yet for this condition." }));
  wrap.appendChild(list);

  return wrap;
}

function renderConditionRows(container, rows, showVersionPerRun) {
  if (!rows.length) {
    container.appendChild(el("p", { className: "hint", text: "No data for this prompt version." }));
    return;
  }
  const byCondition = new Map();
  rows.forEach((r) => {
    if (!byCondition.has(r.condition)) byCondition.set(r.condition, []);
    byCondition.get(r.condition).push(r);
  });
  // Every condition in the canonical CONDITION_ORDER vocabulary always gets a
  // row, even with zero runs -- see renderNotRunConditionRow() above. A
  // condition outside that vocabulary (e.g. "unknown") still only shows up
  // when rows for it actually exist, appended after the canonical ones.
  const extra = Array.from(byCondition.keys())
    .filter((c) => !CONDITION_ORDER.includes(c))
    .sort();
  const conditions = CONDITION_ORDER.concat(extra);
  conditions.forEach((condition) => {
    const runs = byCondition.get(condition);
    if (!runs) {
      container.appendChild(renderNotRunConditionRow(condition));
      return;
    }
    const sorted = runs.slice().sort((a, b) => ((a.date || "") < (b.date || "") ? -1 : 1));
    container.appendChild(renderConditionRow(condition, sorted, showVersionPerRun));
  });
}

function renderConditionsBlock(model, template, rows) {
  const block = el("div", { className: "conditions-block" });
  block.appendChild(el("h3", { className: "conditions-block-title", text: `${template} — ${model}` }));

  const versions = Array.from(new Set(rows.map((r) => r.prompt_version))).sort();
  const defaultVersion = pickDefaultPromptVersion(rows) || versions[0];

  const select = el("select", { className: "select-input conditions-version-select" });
  versions.forEach((v) => select.appendChild(el("option", { text: v, attrs: { value: v } })));
  select.appendChild(el("option", { text: "All versions", attrs: { value: ALL_VERSIONS } }));
  select.value = defaultVersion;

  const picker = el("label", { className: "conditions-version-picker" }, [
    el("span", { text: "Prompt version:" }),
  ]);
  picker.appendChild(select);
  block.appendChild(picker);

  const body = el("div", { className: "conditions-block-body" });
  block.appendChild(body);

  function render() {
    body.replaceChildren();
    const selected = select.value;
    const showingAll = selected === ALL_VERSIONS;
    // Filtering by exact prompt_version equality is what guarantees the
    // single-version view never combines rows from a different prompt
    // version into the same bar/marker set -- "All versions" is the only
    // branch that skips this filter, and it never pools either (every run
    // still gets its own marker); it only adds the mismatch warning below.
    const visible = showingAll ? rows : rows.filter((r) => r.prompt_version === selected);

    if (showingAll && versions.length > 1) {
      body.appendChild(
        el("p", {
          className: "error-state conditions-mismatch-warning",
          text:
            "Showing all prompt versions (" +
            versions.join(", ") +
            ") — runs from different versions are not directly comparable.",
        })
      );
    }

    renderConditionRows(body, visible, showingAll);
  }

  select.addEventListener("change", render);
  render();

  return block;
}

function renderConditionsBlocks(rows) {
  const container = qs("conditions-blocks");
  container.replaceChildren();
  const groups = groupByModelTemplate(rows);
  if (!groups.length) {
    qs("conditions-empty").classList.remove("hidden");
    return;
  }
  groups.forEach((g) => {
    container.appendChild(renderConditionsBlock(g.model, g.template, g.rows));
  });
}

async function init() {
  const { body } = await getJSON("/api/results");
  qs("results-meta").textContent = `Reading from ${body.results_dir}`;

  if (body.errors && body.errors.length) {
    const box = qs("results-errors");
    box.textContent = body.errors.join(" · ");
    box.classList.remove("hidden");
  }

  const rows = body.rows || [];
  if (!rows.length) {
    qs("results-empty").classList.remove("hidden");
    return;
  }

  renderClassifierBanner(rows);
  wireOlderRunsToggle(rows);
  renderResultsTable(rows);
  qs("results-table").classList.remove("hidden");

  renderConditionsBlocks(rows);
}

init();
