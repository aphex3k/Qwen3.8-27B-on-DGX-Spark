"use strict";

// Qwen3.8-27B on DGX Spark — recipe configurator.
// State lives in the URL hash; the command is rendered from data.json.
// No framework, no build step, no external assets.

const AXIS_ORDER = ["tier", "spec", "draft", "ssm", "mem", "path"];

let DATA = null;
let state = {};
let fmt = "docker";

function esc(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}

// ---- state / URL hash --------------------------------------------------------

function stateKey() {
  // cells are keyed without the serving path (same flags either way)
  return ["tier", "spec", "draft", "ssm", "mem"].map(function (a) { return state[a]; }).join("|");
}

function initState() {
  state = Object.assign({}, DATA.defaultState);
  const h = location.hash.replace(/^#/, "");
  if (h) {
    h.split("&").forEach(function (part) {
      const i = part.indexOf("=");
      if (i < 0) return;
      const k = decodeURIComponent(part.slice(0, i));
      const v = decodeURIComponent(part.slice(i + 1));
      if (DATA.axes[k] && (v in DATA.axes[k].options)) state[k] = v;
    });
  }
}

function writeHash() {
  const h = AXIS_ORDER.map(function (a) {
    return a + "=" + encodeURIComponent(state[a]);
  }).join("&");
  history.replaceState(null, "", "#" + h);
  const share = document.getElementById("shareUrl");
  if (share) share.textContent = location.href;
}

// ---- flag assembly -----------------------------------------------------------

// Compute the effective sglang_flags for an arbitrary state, from the shipped
// base set: tier swaps the three co-limiting concurrency flags, ssm/mem/draft
// override their values, spec=off drops the whole DFLASH block.
function flagsFor(s) {
  const overrides = {};
  const tierOpts = DATA.axes.tier.options[s.tier];
  Object.keys(tierOpts.flags).forEach(function (k) { overrides[k] = tierOpts.flags[k]; });
  overrides["--mamba-ssm-dtype"] = s.ssm;
  overrides["--mem-fraction-static"] = s.mem;
  if (s.spec === "on") overrides["--speculative-num-draft-tokens"] = s.draft;
  const drop = s.spec === "off" ? new Set(DATA.specOffFlags) : new Set();
  return DATA.baseFlags
    .filter(function (f) { return !drop.has(f[0]); })
    .map(function (f) { return f.slice(); })
    .map(function (f) {
      if (f[0] in overrides) f[1] = overrides[f[0]];
      return f;
    });
}

function dockerCommand() {
  const L = [];
  L.push('mkdir -p "$HOME/.cache/huggingface" "$HOME/.triton"');
  L.push("docker rm -f qwen38-sglang 2>/dev/null");
  L.push("docker run -d --name qwen38-sglang \\");
  L.push("  --network host --ipc host --privileged --gpus all --shm-size 32g \\");
  L.push("  --cpuset-cpus " + DATA.pinned.cpuset + " \\");
  L.push("  -e HF_HOME=/root/.cache/huggingface \\");
  L.push("  -e TRITON_CACHE_DIR=/root/.triton \\");
  L.push("  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK \\");
  L.push('  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\');
  L.push('  -v "$HOME/.triton:/root/.triton" \\');
  L.push("  " + DATA.pinned.image + " \\");
  L.push("  python3 -m sglang.launch_server \\");
  const flags = flagsFor(state);
  flags.forEach(function (f, i) {
    const text = "    " + f[0] + (f.length > 1 ? " " + f[1] : "");
    L.push(text + (i < flags.length - 1 ? " \\" : ""));
  });
  return L.join("\n");
}

function pythonCommand() {
  const L = ["python3 -m sglang.launch_server \\"];
  const flags = flagsFor(state);
  flags.forEach(function (f, i) {
    const text = "  " + f[0] + (f.length > 1 ? " " + f[1] : "");
    L.push(text + (i < flags.length - 1 ? " \\" : ""));
  });
  return L.join("\n");
}

// ---- diff vs. the shipped recipe ---------------------------------------------

function diffChips() {
  const shipped = flagsFor(DATA.shippedState);
  const current = flagsFor(state);
  const shippedMap = {};
  shipped.forEach(function (f) { shippedMap[f[0]] = f.length > 1 ? f[1] : ""; });
  const currentMap = {};
  current.forEach(function (f) { currentMap[f[0]] = f.length > 1 ? f[1] : ""; });
  const chips = [];
  current.forEach(function (f) {
    const name = f[0];
    const val = f.length > 1 ? f[1] : "";
    if (!(name in shippedMap)) {
      chips.push({ cls: "add", text: "+ " + name + (val ? " " + val : "") });
    } else if (shippedMap[name] !== val) {
      chips.push({ cls: "mod", text: "~ " + name + " " + shippedMap[name] + " → " + val });
    }
  });
  shipped.forEach(function (f) {
    if (!(f[0] in currentMap)) chips.push({ cls: "del", text: "− " + f[0] });
  });
  return chips;
}

function renderDiff() {
  const host = document.getElementById("diff");
  if (!host) return;
  const chips = diffChips();
  if (chips.length === 0) {
    host.innerHTML = '<div class="diff empty">No flags differ from the shipped recipe.</div>';
    return;
  }
  host.innerHTML = '<div class="diff"><span class="diff-title">vs. shipped recipe</span>'
    + chips.map(function (c) {
        return '<span class="chip ' + c.cls + '">' + esc(c.text) + "</span>";
      }).join("")
    + "</div>";
}

// ---- measured-cell badge + warnings ------------------------------------------

function badgeInfo() {
  const c = DATA.cells[stateKey()];
  if (!c) return { cls: "unverified", text: "Unverified — best-effort combo" };
  const bits = [];
  if (c.single) bits.push(c.single + " tok/s single");
  if (c.agg16) bits.push(c.agg16 + " tok/s @16 streams");
  if (c.ttftMs) bits.push(c.ttftMs + " ms TTFT");
  if (c.freeGb) bits.push(c.freeGb + " GB free");
  return {
    cls: "verified",
    text: "Measured" + (bits.length ? " · " + bits.join(" · ") : "")
  };
}

function activeWarnings() {
  const w = [];
  if (state.mem === "0.90") w.push(DATA.warnings.memHigh);
  if (state.ssm === "float32") w.push(DATA.warnings.ssmFloat32);
  if (state.spec === "off") w.push(DATA.warnings.specOff);
  if (state.path === "standalone" && state.spec === "on") w.push(DATA.warnings.offline);
  if (state.tier === "high-throughput") w.push(DATA.warnings.htPool);
  return w;
}

function renderWarnings() {
  const host = document.getElementById("warnings");
  if (!host) return;
  const w = activeWarnings();
  host.innerHTML = w.map(function (text) {
    return '<div class="warning"><span class="wmark">!</span>' + esc(text) + "</div>";
  }).join("");
}

// ---- SparkStation variant -----------------------------------------------------

function renderSparkBody() {
  const sp = DATA.sparkstation;
  const tierOpts = DATA.axes.tier.options[state.tier];
  const shipped = DATA.shippedState;

  const flagRows = [];
  if (state.tier !== "interactive") {
    Object.keys(tierOpts.flags).forEach(function (k) {
      flagRows.push([k, tierOpts.flags[k]]);
    });
  }
  if (state.spec === "off") {
    // removing the DFLASH block from models.yaml is a bigger edit; say so
    flagRows.push(["# remove the five --speculative-* flags", ""]);
  } else if (state.draft !== shipped.draft) {
    flagRows.push(["--speculative-num-draft-tokens", state.draft]);
  }
  if (state.ssm !== "bfloat16") flagRows.push(["--mamba-ssm-dtype", state.ssm]);

  const yaml = ["# models.yaml — qwen3.8-sglang entry (back up first: cp models.yaml models.yaml.bak)",
    "qwen3.8-sglang:"];
  yaml.push('  host: primary                      # ships pinned to worker1');
  const extra = ["  extra_args:"];
  if (state.mem !== "0.82") {
    extra.push("    mem_fraction_static: " + state.mem + "   # needs the mem-fraction override patch; stock launcher clamps at 0.82");
  }
  if (flagRows.length > 0) {
    extra.push("    sglang_flags:                    # set instead of the shipped values");
    flagRows.forEach(function (r) {
      extra.push(r[1] === "" ? "      " + r[0] : '      - "' + r[0] + '" - "' + r[1] + '"');
    });
  } else {
    extra.push("    # no flag changes from the shipped values");
  }
  const yamlText = yaml.concat(extra).join("\n");

  return [
    "<h3>1. Install</h3>",
    '<div class="cmd-box"><pre>' + esc(sp.install.join("\n")) + "</pre><button type=\"button\" class=\"copy-btn\">Copy</button></div>",
    "<h3>2. Edit <code>models.yaml</code></h3>",
    '<div class="cmd-box"><pre>' + esc(yamlText) + "</pre><button type=\"button\" class=\"copy-btn\">Copy</button></div>",
    "<h3>3. Relax the health probe</h3>",
    "<p>In <code>.env</code>, a model saturated on long prefill cannot answer the 5 s probe, so the supervisor kills a healthy model mid-job:</p>",
    '<div class="cmd-box"><pre>HEALTH_CHECK_TIMEOUT_SECONDS=30   # shipped: 5</pre><button type="button" class="copy-btn">Copy</button></div>',
    "<h3>4. Start</h3>",
    '<div class="cmd-box"><pre>' + esc(sp.start) + "</pre><button type=\"button\" class=\"copy-btn\">Copy</button></div>",
    "<p>" + esc(sp.gateway) + "</p>",
    "<h3>Caveats</h3>",
    "<ul class=\"caveats\">" + sp.caveats.map(function (c) { return "<li>" + esc(c) + "</li>"; }).join("") + "</ul>"
  ].join("");
}

// ---- command card --------------------------------------------------------------

function renderCommandCard() {
  const isSpark = state.path === "sparkstation";
  document.getElementById("fmtToggle").style.display = isSpark ? "none" : "";
  document.querySelectorAll("#fmtToggle button").forEach(function (b) {
    b.classList.toggle("on", b.dataset.fmt === fmt);
  });

  const info = badgeInfo();
  const badge = document.getElementById("badge");
  badge.className = "pill " + info.cls;
  badge.textContent = info.text;

  const body = document.getElementById("cmdBody");
  const note = document.getElementById("cmdNote");
  if (isSpark) {
    body.innerHTML = renderSparkBody() + '<div id="warnings"></div>';
    note.textContent = "";
    renderWarnings();
    return;
  }
  const cmd = fmt === "docker" ? dockerCommand() : pythonCommand();
  body.innerHTML =
    '<div class="cmd-box"><pre id="cmdPre"></pre>' +
    '<button type="button" class="copy-btn">Copy</button></div>' +
    "<div id=\"diff\"></div>" +
    "<div id=\"warnings\"></div>";
  document.getElementById("cmdPre").textContent = cmd;
  note.textContent = fmt === "docker"
    ? "Detached container named qwen38-sglang; follow boot with docker logs -f qwen38-sglang. The server answers on :8888 (OpenAI- and Anthropic-compatible)."
    : "For running SGLang on the host or in a container of your own. The weights must already be in the HF cache at the pinned revisions; the first boot downloads ~25 GB.";
  renderDiff();
  renderWarnings();
}

// ---- axis cards ------------------------------------------------------------------

function renderAxisCard(axis) {
  const a = DATA.axes[axis];
  const disabled = axis === "draft" && state.spec === "off";
  const opts = Object.keys(a.options).map(function (key) {
    const o = a.options[key];
    const sel = state[axis] === key ? " on" : "";
    return '<button type="button" class="opt' + sel + '" data-axis="' + axis + '" data-value="' + esc(key) + '">'
      + '<span class="opt-label">' + esc(o.label) + "</span>"
      + (o.hint ? '<span class="opt-hint">' + esc(o.hint) + "</span>" : "")
      + "</button>";
  }).join("");
  return '<div class="card axis' + (disabled ? " disabled" : "") + '">'
    + "<h3>" + esc(a.label) + "</h3>"
    + '<p class="intro">' + esc(a.intro) + "</p>"
    + '<div class="opts">' + opts + "</div>"
    + "</div>";
}

function renderAxes() {
  document.getElementById("axes").innerHTML =
    AXIS_ORDER.map(renderAxisCard).join("");
}

// ---- static sections --------------------------------------------------------------

function renderHero() {
  document.getElementById("heroTitle").textContent = DATA.meta.title;
  document.getElementById("heroLede").textContent = DATA.meta.subtitle;
  document.getElementById("heroHardware").textContent = DATA.meta.hardware;
  document.getElementById("heroRepro").innerHTML = esc(DATA.meta.reproduced)
    + ' — <a href="' + esc(DATA.repo.url) + '">recipe, traps and full measurements in the repo</a>.';
  document.getElementById("headlines").innerHTML = DATA.headlines.map(function (h) {
    return '<div class="hl"><span class="hl-value">' + esc(h.value) + "</span>"
      + "<span>" + esc(h.label) + "</span>"
      + '<span class="hl-note">' + esc(h.note) + "</span></div>";
  }).join("");
}

function renderVerify() {
  const v = DATA.verify;
  document.getElementById("verifyBody").innerHTML =
    "<p>" + esc(v.intro) + "</p>" +
    '<h3>Standalone server (from the docker command above)</h3>' +
    '<div class="cmd-box"><pre>' + esc(v.envStandalone.join("\n")) + "</pre><button type=\"button\" class=\"copy-btn\">Copy</button></div>" +
    '<h3>SparkStation gateway</h3>' +
    '<div class="cmd-box"><pre>' + esc(v.envGateway.join("\n")) + "</pre><button type=\"button\" class=\"copy-btn\">Copy</button></div>" +
    '<h3>Run the benches</h3>' +
    '<div class="cmd-box"><pre>' + esc(v.run.join("\n")) + "</pre><button type=\"button\" class=\"copy-btn\">Copy</button></div>";
}

function renderFooter() {
  const p = DATA.pinned;
  document.getElementById("footerBody").innerHTML =
    "<div>Pinned inputs: <code>" + esc(p.image) + "</code> · target " +
    esc(p.targetRepo) + " @ <code>" + esc(p.targetRevision.slice(0, 8)) + "</code> · "
    + "draft " + esc(p.draftRepo) + " @ <code>" + esc(p.draftRevision.slice(0, 8)) + "</code> · "
    + esc(p.toolkit) + "</div>" +
    "<div>" + p.notes.map(function (n) { return esc(n); }).join(" ") + "</div>" +
    '<div>Full measurement data: <a href="' + esc(DATA.repo.url) + "/tree/main/" + esc(DATA.repo.resultsUrl) + '">'
    + esc(DATA.repo.resultsUrl) + "</a>. Found a combo that disagrees with your numbers? "
    + '<a href="' + esc(DATA.repo.url) + '/issues">Open an issue</a> — the green badge marks the cells this repo actually measured.</div>';
}

// ---- events ------------------------------------------------------------------------

function bindEvents() {
  document.getElementById("axes").addEventListener("click", function (e) {
    const btn = e.target.closest("button.opt");
    if (!btn) return;
    const axis = btn.dataset.axis;
    const value = btn.dataset.value;
    if (axis === "draft" && state.spec === "off") return;
    if (state[axis] === value) return;
    state[axis] = value;
    renderAxes();
    renderCommandCard();
    writeHash();
  });

  document.getElementById("fmtToggle").addEventListener("click", function (e) {
    const btn = e.target.closest("button[data-fmt]");
    if (!btn || state.path === "sparkstation") return;
    if (fmt === btn.dataset.fmt) return;
    fmt = btn.dataset.fmt;
    renderCommandCard();
  });

  document.body.addEventListener("click", function (e) {
    const btn = e.target.closest(".copy-btn");
    if (!btn) return;
    const pre = btn.parentElement.querySelector("pre");
    if (!pre) return;
    copyText(pre.textContent, btn);
  });

  window.addEventListener("hashchange", function () {
    initState();
    renderAxes();
    renderCommandCard();
    writeHash();
  });
}

function copyText(text, btn) {
  const done = function () {
    btn.textContent = "Copied";
    setTimeout(function () { btn.textContent = "Copy"; }, 1500);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, function () {
      fallbackCopy(text, done);
    });
  } else {
    fallbackCopy(text, done);
  }
}

function fallbackCopy(text, done) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); } catch (err) { /* no clipboard */ }
  document.body.removeChild(ta);
  done();
}

// ---- boot -----------------------------------------------------------------------------

function renderAll() {
  renderAxes();
  renderCommandCard();
  writeHash();
}

window.addEventListener("DOMContentLoaded", function () {
  fetch("assets/data.json")
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(function (data) {
      DATA = data;
      initState();
      bindEvents();
      renderHero();
      renderAxes();
      renderCommandCard();
      renderVerify();
      renderFooter();
      writeHash();
      document.getElementById("loadError").hidden = true;
    })
    .catch(function (err) {
      const el = document.getElementById("loadError");
      el.hidden = false;
      el.textContent = "Could not load assets/data.json (" + err.message + "). Open the site over HTTP (GitHub Pages, or: cd docs && python3 -m http.server).";
    });
});
