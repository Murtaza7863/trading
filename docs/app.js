const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const STORE = "trading-desk-journal-v1";
const BAN = {
  CBRX: "New 2x product — skip overnight",
  CBRZ: "New 2x product — skip overnight",
  CBRS: "Recent IPO — skip",
  SKUU: "New 2x product — skip overnight",
  SOXL: "3x daily-reset — do not hold overnight",
  SOXS: "3x daily-reset — do not hold overnight",
  UVXY: "VIX path product — skip",
};

let apiMode = false;
let clock = {};

function money(n) {
  if (n == null || Number.isNaN(n)) return "—";
  const abs = Math.abs(n).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  });
  return (n < 0 ? "−$" : "+$") + abs;
}
function pct(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
}
function num(n, d = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toFixed(d);
}
function clsPnl(n) {
  if (n == null) return "";
  return n > 0 ? "up" : n < 0 ? "down" : "";
}
function flash(msg) {
  const el = $("#flash");
  el.hidden = !msg;
  el.textContent = msg || "";
}

function pageBase() {
  const href = window.location.href.split("#")[0].split("?")[0];
  if (href.endsWith(".html")) {
    return href.slice(0, href.lastIndexOf("/") + 1);
  }
  return href.endsWith("/") ? href : `${href}/`;
}
function asset(path) {
  return new URL(path, pageBase()).toString();
}

function etClock() {
  const now = new Date();
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    weekday: "short",
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(now);
  const grab = (t) => parts.find((p) => p.type === t)?.value;
  const hour = Number(grab("hour"));
  const minute = Number(grab("minute"));
  const y = grab("year");
  const months = {
    Jan: "01",
    Feb: "02",
    Mar: "03",
    Apr: "04",
    May: "05",
    Jun: "06",
    Jul: "07",
    Aug: "08",
    Sep: "09",
    Oct: "10",
    Nov: "11",
    Dec: "12",
  };
  const m = months[grab("month")];
  const d = grab("day").padStart(2, "0");
  const hh = String(hour).padStart(2, "0");
  const mm = String(minute).padStart(2, "0");
  const mins = hour * 60 + minute;
  const inRth = mins >= 9 * 60 + 30 && mins < 16 * 60;
  const wd = grab("weekday");
  const weekend = wd === "Sat" || wd === "Sun";
  let session = "overnight";
  if (weekend) session = "weekend";
  else if (mins >= 4 * 60 && mins < 9 * 60 + 30) session = "premarket";
  else if (inRth) session = "rth";
  else if (mins >= 16 * 60 && mins < 20 * 60) session = "afterhours";
  return {
    et_local: `${y}-${m}-${d}T${hh}:${mm}`,
    et_label: `${grab("weekday")} ${d} ${grab("month")} ${y}, ${hh}:${mm} ET`,
    session,
    after_close:
      session === "afterhours" ||
      session === "overnight" ||
      session === "weekend",
    morning_grind: inRth && hour >= 10 && hour < 12,
    premarket: session === "premarket",
    afterhours: session === "afterhours",
    weekend,
  };
}

function clockNote(c) {
  if (c.session === "premarket")
    return "Premarket 4:00–9:30 ET. Use PM fuel to look. Do not hold 2x through the open just because PM is green.";
  if (c.session === "afterhours")
    return "Night session 16:00–20:00 ET. Flatten 2x — overnight is where the old book got hurt.";
  if (c.session === "overnight" || c.session === "weekend")
    return "Cash and night tape are closed. Board is the last session. Do not carry 2x.";
  if (c.morning_grind)
    return "10:00–12:00 ET was the weakest window on the old book.";
  return "Cash session. Same-day only. Flatten before 16:00 ET.";
}

function loadStore() {
  try {
    return JSON.parse(localStorage.getItem(STORE) || "[]");
  } catch {
    return [];
  }
}
function saveStore(trades) {
  localStorage.setItem(STORE, JSON.stringify(trades));
}

function annotate(trade, others) {
  const t = { ...trade };
  const qty = Number(t.qty) || 0;
  const pxIn = Number(t.entry_price) || 0;
  const side = (t.side || "buy").toLowerCase();
  const sign = side === "buy" ? 1 : -1;
  t.symbol = String(t.symbol || "").toUpperCase();
  t.entry_notional = qty * pxIn;
  t.status =
    t.exit_price != null && t.exit_price !== "" && t.exit_time
      ? "closed"
      : "open";
  const entryDay = (t.entry_time || "").slice(0, 10);
  const exitDay = (t.exit_time || "").slice(0, 10);
  const entryHour = Number((t.entry_time || "").slice(11, 13));
  if (t.status === "closed") {
    const pxOut = Number(t.exit_price);
    t.pnl = (pxOut - pxIn) * qty * sign;
    t.return_pct = pxIn ? (pxOut / pxIn - 1) * 100 * sign : null;
    const exitHour = Number((t.exit_time || "").slice(11, 13));
    t.overnight = entryDay !== exitDay || (entryHour < 16 && exitHour >= 16);
  } else {
    t.pnl = null;
    const today = etClock().et_local.slice(0, 10);
    t.overnight =
      entryDay !== today || Number(etClock().et_local.slice(11, 13)) >= 16;
  }
  t.same_session = !t.overnight;
  const warnings = [];
  if (BAN[t.symbol]) warnings.push(BAN[t.symbol]);
  if (t.overnight)
    warnings.push(
      "Overnight 2x/3x is where the old book got hurt. Flatten before 16:00 ET.",
    );
  if (entryHour >= 10 && entryHour < 12)
    warnings.push("10:00–12:00 ET was a weak window historically.");
  const otherOpen = others.some(
    (o) =>
      o.id !== t.id &&
      String(o.symbol).toUpperCase() === t.symbol &&
      !o.exit_time,
  );
  if (otherOpen)
    warnings.push("Another open lot in this name — do not average down.");
  t.warnings = warnings;
  return t;
}

function journalState() {
  const raw = loadStore();
  const trades = raw
    .map((t) =>
      annotate(
        t,
        raw.filter((x) => x.id !== t.id),
      ),
    )
    .sort((a, b) => String(b.entry_time).localeCompare(String(a.entry_time)));
  const closed = trades.filter((t) => t.status === "closed" && t.pnl != null);
  const opens = trades.filter((t) => t.status === "open");
  const wins = closed.filter((t) => t.pnl > 0);
  const losses = closed.filter((t) => t.pnl <= 0);
  const same = closed.filter((t) => t.same_session);
  const overnight = closed.filter((t) => t.overnight);
  return {
    trades,
    stats: {
      n_open: opens.length,
      n_closed: closed.length,
      open_notional: opens.reduce((s, t) => s + (t.entry_notional || 0), 0),
      realized_pnl: closed.reduce((s, t) => s + t.pnl, 0),
      same_session_pnl: same.reduce((s, t) => s + t.pnl, 0),
      overnight_pnl: overnight.reduce((s, t) => s + t.pnl, 0),
      win_rate: closed.length ? wins.length / closed.length : null,
      avg_win: wins.length
        ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length
        : null,
      avg_loss: losses.length
        ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length
        : null,
    },
  };
}

async function api(path, opts = {}) {
  const ctrl = new AbortController();
  const ms = opts.timeoutMs || 20000;
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    const res = await fetch(asset(path), {
      method: opts.method || "GET",
      headers: { "Content-Type": "application/json" },
      body: opts.body ? JSON.stringify(opts.body) : undefined,
      signal: ctrl.signal,
      cache: "no-store",
    });
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { detail: text };
    }
    if (!res.ok) {
      const detail = data && data.detail;
      throw new Error(
        typeof detail === "string" && detail
          ? detail
          : res.statusText || "Request failed",
      );
    }
    return data;
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new Error("Timed out waiting for the board.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function loadSnapshot(bust) {
  const q = bust ? `?t=${Date.now()}` : "";
  const urls = [asset(`watchlist.json${q}`)];
  if (location.hostname.endsWith("github.io")) {
    urls.push(
      `https://raw.githubusercontent.com/Murtaza7863/trading/main/docs/watchlist.json${q}`,
    );
  }
  let lastErr;
  for (const url of urls) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(res.statusText);
      return await res.json();
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr || new Error("Could not load the volatility board.");
}

function renderStats(s = {}) {
  const cells = [
    ["Open lots", s.n_open ?? 0, ""],
    [
      "Open notional",
      s.open_notional != null ? "$" + Number(s.open_notional).toFixed(0) : "—",
      "",
    ],
    ["Journal P&L", money(s.realized_pnl), clsPnl(s.realized_pnl)],
    [
      "Same-session / overnight",
      `${money(s.same_session_pnl)} / ${money(s.overnight_pnl)}`,
      "",
    ],
  ];
  $("#stats").innerHTML = cells
    .map(
      ([label, val, cls]) =>
        `<div class="stat"><b class="${cls}">${val}</b><span>${label}</span></div>`,
    )
    .join("");
}

function renderOpens(trades) {
  const opens = trades.filter((t) => t.status === "open");
  $("#open-empty").hidden = opens.length > 0;
  $("#opens").innerHTML = opens
    .map((t) => {
      const warns = (t.warnings || [])
        .map((w) => `<span class="pill bad">${w}</span>`)
        .join("");
      const hold = t.overnight
        ? `<span class="pill bad">Overnight / still open after close</span>`
        : `<span class="pill ok">Same session so far</span>`;
      return `<div class="open">
        <strong>${t.symbol}</strong> · ${t.side} ${t.qty} @ ${num(t.entry_price)}
        <div class="meta">${t.entry_time || ""} · notional $${num(t.entry_notional, 0)}</div>
        <div class="warns">${hold}${warns}</div>
        <button class="btn" data-close="${t.id}" data-sym="${t.symbol}">Close lot</button>
      </div>`;
    })
    .join("");
}

function renderJournal(trades) {
  $("#journal-body").innerHTML = trades
    .map((t) => {
      const io = `${num(t.entry_price)} → ${t.exit_price != null ? num(t.exit_price) : "open"}`;
      const hold = t.overnight
        ? "overnight"
        : t.status === "open"
          ? "open"
          : "same day";
      return `<tr>
        <td><strong>${t.symbol}</strong></td>
        <td>${t.side}</td>
        <td class="num">${t.qty}</td>
        <td>${io}<div class="muted">${t.entry_time || ""}</div></td>
        <td class="num ${clsPnl(t.pnl)}">${t.status === "closed" ? money(t.pnl) : "—"}</td>
        <td>${hold}</td>
        <td>${t.notes || ""}</td>
        <td><button class="btn ghost" data-del="${t.id}">Delete</button></td>
      </tr>`;
    })
    .join("");
}

function sessLabel(kind) {
  return (
    {
      premarket: "Premarket",
      rth: "Cash",
      afterhours: "Night tape",
      overnight: "Closed",
      weekend: "Weekend",
    }[kind] || "—"
  );
}

function sessCell(ret, rng, extra) {
  if (ret == null && rng == null) {
    return extra ? `<span class="muted">${extra}</span>` : "—";
  }
  const cls = ret > 0 ? "up" : ret < 0 ? "down" : "";
  return `<span class="${cls}">${pct(ret)}</span>${extra ? `<div class="muted">${extra}</div>` : ""}`;
}

function leanCell(r) {
  const lean = r.lean || "flat";
  const cls =
    lean === "up" ? "ok" : lean === "down" ? "bad" : lean === "mixed" ? "" : "";
  const arrow =
    lean === "up" ? "↑" : lean === "down" ? "↓" : lean === "mixed" ? "↕" : "·";
  const label =
    lean === "up"
      ? "Up"
      : lean === "down"
        ? "Down"
        : lean === "mixed"
          ? "Mixed"
          : "Flat";
  const strong = r.lean_strength === 2 ? " strong" : "";
  return `<span class="pill ${cls}${strong}">${arrow} ${label}</span><div class="muted">${r.lean_why || "tape read, not a signal"}</div>`;
}

function splitBar(r) {
  const pm = Number(r.split_pm) || 0;
  const rth = Number(r.split_rth) || 0;
  const ah = Number(r.split_ah) || 0;
  const tot = pm + rth + ah;
  if (tot <= 0) return "";
  const w = (x) => Math.max(4, Math.round((x / tot) * 72));
  return `<div class="split" title="Today: premarket / cash / night"><i class="pm" style="width:${w(pm)}px"></i><i class="rth" style="width:${w(rth)}px"></i><i class="ah" style="width:${w(ah)}px"></i></div>`;
}

function renderWatchlist(data) {
  const rows = data.rows || [];
  $("#wl-meta").textContent = data.generated
    ? `Board from ${data.generated}. Fuel 0–10 = how much of a normal day has printed today.`
    : "No board file yet.";
  $("#tape").textContent = data.tape || "";
  const legend = $("#fuel-legend");
  if (legend) {
    legend.textContent =
      data.fuel_legend ||
      "Fuel is today's printed move vs ATR. Last night is the Night column, not the rank. Lean is a tape read — fade comes back mixed. Not a forecast.";
  }
  $("#sym-list").innerHTML = rows
    .flatMap((r) => [r.ticker, r.vehicle].filter(Boolean))
    .map((s) => `<option value="${s}"></option>`)
    .join("");
  $("#wl-body").innerHTML = rows
    .map((r, i) => {
      const fuel = r.fuel_score;
      const w = fuel != null ? Math.round((Math.min(fuel, 10) / 10) * 72) : 8;
      const flags = [];
      if (r.days_to_earnings === 0)
        flags.push(`<span class="pill bad">Print today</span>`);
      else if (r.earnings_soon)
        flags.push(`<span class="pill bad">Earnings soon</span>`);
      if (r.spent) flags.push(`<span class="pill">Spent</span>`);
      if (r.setup === "continuation")
        flags.push(
          `<span class="pill ${r.lean === "down" ? "bad" : "ok"}">Continuation</span>`,
        );
      if (r.setup === "fade") flags.push(`<span class="pill">Fade</span>`);
      if (r.ok_vehicle) flags.push(`<span class="pill ok">${r.vehicle}</span>`);
      else flags.push(`<span class="pill">Underlying</span>`);
      const nightExtra =
        r.night_from === "last_night"
          ? "last night"
          : r.night_from === "ah"
            ? "AH"
            : "";
      const atr = r.atr14_pct;
      const leftPct = r.atr_left;
      let left = "—";
      if (leftPct != null && Number.isFinite(Number(leftPct))) {
        const sub =
          atr != null
            ? r.spent
              ? "spent"
              : `of ${Number(atr).toFixed(1)} ATR`
            : "";
        left = `${Number(leftPct).toFixed(1)}%${sub ? `<div class="muted">${sub}</div>` : ""}`;
      }
      const hot = i < 3 ? " hot" : "";
      return `<tr class="${hot.trim()}" data-sym="${r.ok_vehicle ? r.vehicle : r.ticker}">
        <td class="rank">${i + 1}</td>
        <td><strong>${r.ticker}</strong><div class="muted">${r.name}</div></td>
        <td class="num"><span class="fuel" style="width:${w}px"></span><span class="fuel-n">${num(fuel, 1)}</span>${splitBar(r)}<div class="muted">${r.fuel_note || ""}</div></td>
        <td>${leanCell(r)}</td>
        <td class="num">${num(r.last, 2)}</td>
        <td class="num">${sessCell(r.pm_ret_pct)}</td>
        <td class="num">${sessCell(r.rth_ret_pct)}</td>
        <td class="num">${sessCell(r.ah_ret_pct, null, nightExtra)}</td>
        <td class="num">${left}</td>
        <td>${flags.join(" ")}</td>
      </tr>`;
    })
    .join("");
}

function renderBook(lots) {
  $("#book-body").innerHTML = (lots || [])
    .map((t) => {
      const ov = String(t.overnight) === "True" || t.overnight === true;
      return `<tr>
        <td><strong>${t.symbol}</strong></td>
        <td class="num">${t.qty}</td>
        <td>${num(t.entry_price)} → ${num(t.exit_price)}<div class="muted">${t.entry_time || ""}</div></td>
        <td>${ov ? "Yes" : "No"}</td>
        <td class="num ${clsPnl(t.pnl)}">${money(t.pnl)}</td>
        <td>${t.entry_session || ""}</td>
      </tr>`;
    })
    .join("");
}

function paintJournal(jn) {
  renderStats(jn.stats);
  renderOpens(jn.trades);
  renderJournal(jn.trades);
}

async function loadAll() {
  clock = etClock();
  try {
    const clk = await api("api/clock");
    apiMode = true;
    clock = clk;
  } catch {
    apiMode = false;
  }

  $("#clock").textContent = clock.et_label;
  const pill = $("#session-pill");
  if (pill) {
    pill.textContent = sessLabel(clock.session);
    pill.className =
      "pill" +
      (clock.session === "rth"
        ? " ok"
        : clock.session === "premarket" || clock.session === "afterhours"
          ? ""
          : " bad");
  }
  $("#clock-note").textContent = clockNote(clock);
  $("#log-form").entry_time.value = clock.et_local;
  $("#tab-book").hidden = !apiMode;
  $("#refresh").hidden = false;
  $("#refresh").textContent = "Refresh board";

  let wl = { rows: [] };
  try {
    wl = apiMode ? await api("api/watchlist") : await loadSnapshot(false);
  } catch {
    flash("Could not load the volatility board.");
  }
  renderWatchlist(wl);

  if (apiMode) {
    const jn = await api("api/journal");
    paintJournal(jn);
    const hist = await api("api/history");
    renderBook(hist.lots || []);
  } else {
    paintJournal(journalState());
  }
}

$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.toggle("on", b === btn));
    $$(".panel").forEach((p) =>
      p.classList.toggle("on", p.dataset.panel === btn.dataset.tab),
    );
  });
});

$("#log-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = Object.fromEntries(new FormData(e.target).entries());
  try {
    if (apiMode) {
      await api("api/journal", { method: "POST", body });
    } else {
      const trades = loadStore();
      trades.push({
        id: Math.random().toString(36).slice(2, 10),
        symbol: body.symbol,
        side: body.side,
        qty: Number(body.qty),
        entry_price: Number(body.entry_price),
        entry_time: body.entry_time,
        notes: body.notes || "",
      });
      saveStore(trades);
    }
    flash("");
    e.target.reset();
    e.target.entry_time.value = clock.et_local;
    await loadAll();
  } catch (err) {
    flash(err.message);
  }
});

$("#refresh").addEventListener("click", async () => {
  const btn = $("#refresh");
  btn.disabled = true;
  btn.textContent = apiMode ? "Pulling Yahoo…" : "Reloading board…";
  flash("");
  try {
    if (apiMode) {
      renderWatchlist(
        await api("api/watchlist/refresh", {
          method: "POST",
          body: {},
          timeoutMs: 90000,
        }),
      );
    } else {
      const data = await loadSnapshot(true);
      renderWatchlist(data);
      flash(
        data.generated
          ? `Board snapshot from ${data.generated}. Live Yahoo needs the local desk.`
          : "Loaded the committed board snapshot.",
      );
    }
  } catch (err) {
    flash(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh board";
  }
});

document.addEventListener("click", (e) => {
  const row = e.target.closest("tr[data-sym]");
  if (row && !e.target.closest("button"))
    $("#log-form").symbol.value = row.dataset.sym;
  const closeBtn = e.target.closest("[data-close]");
  if (closeBtn) {
    $("#close-sym").textContent = closeBtn.dataset.sym;
    $("#close-form").id.value = closeBtn.dataset.close;
    $("#close-form").exit_time.value = clock.et_local;
    $("#close-dlg").showModal();
  }
  const del = e.target.closest("[data-del]");
  if (del) {
    if (apiMode) {
      api(`api/journal/${del.dataset.del}`, { method: "DELETE" })
        .then(loadAll)
        .catch((err) => flash(err.message));
    } else {
      saveStore(loadStore().filter((t) => t.id !== del.dataset.del));
      loadAll();
    }
  }
});

$("#close-cancel").addEventListener("click", () => $("#close-dlg").close());
$("#close-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const id = fd.get("id");
  try {
    if (apiMode) {
      await api(`api/journal/${id}/close`, {
        method: "POST",
        body: {
          exit_price: fd.get("exit_price"),
          exit_time: fd.get("exit_time"),
        },
      });
    } else {
      const trades = loadStore().map((t) =>
        t.id === id
          ? {
              ...t,
              exit_price: Number(fd.get("exit_price")),
              exit_time: fd.get("exit_time"),
            }
          : t,
      );
      saveStore(trades);
    }
    $("#close-dlg").close();
    await loadAll();
  } catch (err) {
    flash(err.message);
  }
});

loadAll().catch((err) => flash(err.message));
