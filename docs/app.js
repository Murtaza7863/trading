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
  return {
    et_local: `${y}-${m}-${d}T${hh}:${mm}`,
    et_label: `${grab("weekday")} ${d} ${grab("month")} ${y}, ${hh}:${mm} ET`,
    after_close: !inRth,
    morning_grind: hour >= 10 && hour < 12,
  };
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
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
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

function renderWatchlist(data) {
  const rows = data.rows || [];
  $("#wl-meta").textContent = data.generated
    ? `Board from ${data.generated}. Rank is fuel, not a buy.`
    : "No board file yet.";
  $("#tape").textContent = data.tape || "";
  $("#sym-list").innerHTML = rows
    .flatMap((r) => [r.ticker, r.vehicle].filter(Boolean))
    .map((s) => `<option value="${s}"></option>`)
    .join("");
  const maxFuel = Math.max(...rows.map((r) => r.fuel_score || 0), 1);
  $("#wl-body").innerHTML = rows
    .map((r) => {
      const w = r.fuel_score ? Math.round((r.fuel_score / maxFuel) * 72) : 8;
      const flags = [];
      if (r.earnings_soon)
        flags.push(`<span class="pill bad">Earnings soon</span>`);
      if (r.ok_vehicle) flags.push(`<span class="pill ok">${r.vehicle}</span>`);
      else flags.push(`<span class="pill">Underlying only</span>`);
      const retCls = r.ret_1d_pct > 0 ? "up" : r.ret_1d_pct < 0 ? "down" : "";
      return `<tr data-sym="${r.ok_vehicle ? r.vehicle : r.ticker}">
        <td><strong>${r.ticker}</strong><div class="muted">${r.name} · ${r.group}</div></td>
        <td class="num"><span class="fuel" style="width:${w}px"></span>${num(r.fuel_score)}</td>
        <td class="num">${num(r.last, 2)}</td>
        <td class="num ${retCls}">${pct(r.ret_1d_pct)}</td>
        <td class="num">${pct(r.ret_5d_pct)}</td>
        <td class="num">${num(r.vol_20d_ann_pct, 1)}%</td>
        <td class="num">${num(r.atr14_pct, 2)}%</td>
        <td class="num">${num(r.today_range_pct, 2)}%</td>
        <td>${r.ok_vehicle ? r.vehicle : "—"}</td>
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
  $("#clock-note").textContent = clock.after_close
    ? "Cash session is closed. Do not carry 2x/3x overnight."
    : clock.morning_grind
      ? "10:00–12:00 ET was the weakest window on the old book."
      : "Times are America/New_York.";
  $("#log-form").entry_time.value = clock.et_local;
  $("#tab-book").hidden = !apiMode;
  $("#refresh").hidden = !apiMode;
  if (!apiMode) {
    $("#refresh").textContent = "Snapshot board";
  }

  let wl = { rows: [] };
  try {
    wl = apiMode
      ? await api("api/watchlist")
      : await (await fetch("watchlist.json")).json();
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
  if (!apiMode) return;
  const btn = $("#refresh");
  btn.disabled = true;
  btn.textContent = "Pulling Yahoo (~30s)…";
  try {
    renderWatchlist(
      await api("api/watchlist/refresh", { method: "POST", body: {} }),
    );
  } catch (err) {
    flash(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh from Yahoo";
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
