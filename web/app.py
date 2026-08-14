"""Local trading desk. Watchlist + journal. Not a live broker and not a bot."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import WATCHLIST_SKIP  # noqa: E402
from src.journal import (  # noqa: E402
    add_trade,
    broker_history,
    close_trade,
    delete_trade,
    et_clock,
    list_trades,
    seed_open_from_broker,
    stats,
    update_trade,
)
from src.watchlist import TAPE_NOTE, build_watchlist, load_cached_watchlist, records, write_watchlist  # noqa: E402

DOCS = ROOT / "docs"

app = FastAPI(title="Trading desk", docs_url=None, redoc_url=None)


class TradeIn(BaseModel):
    symbol: str
    side: str = "buy"
    qty: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    entry_time: str | None = None
    notes: str = ""


class CloseIn(BaseModel):
    exit_price: float = Field(gt=0)
    exit_time: str | None = None
    notes: str | None = None


class TradePatch(BaseModel):
    symbol: str | None = None
    side: str | None = None
    qty: float | None = None
    entry_price: float | None = None
    entry_time: str | None = None
    exit_price: float | None = None
    exit_time: str | None = None
    notes: str | None = None


@app.get("/")
def index():
    return FileResponse(DOCS / "index.html")


@app.get("/styles.css")
def css():
    return FileResponse(DOCS / "styles.css")


@app.get("/app.js")
def js():
    return FileResponse(DOCS / "app.js", media_type="text/javascript")


@app.get("/watchlist.json")
def watchlist_file():
    path = DOCS / "watchlist.json"
    if not path.exists():
        raise HTTPException(404, "No watchlist snapshot")
    return FileResponse(path)


@app.get("/api/clock")
def clock():
    return et_clock()


@app.get("/api/watchlist")
def watchlist():
    cached = load_cached_watchlist()
    return {
        **cached,
        "tape": TAPE_NOTE,
        "ban": WATCHLIST_SKIP,
    }


@app.post("/api/watchlist/refresh")
def refresh_watchlist():
    df = build_watchlist()
    write_watchlist(df, tape_note=TAPE_NOTE)
    return {
        "generated": load_cached_watchlist().get("generated"),
        "rows": records(df),
        "tape": TAPE_NOTE,
        "ban": WATCHLIST_SKIP,
    }


@app.get("/api/journal")
def journal():
    seed_open_from_broker()
    return {"trades": list_trades(), "stats": stats()}


@app.post("/api/journal")
def journal_add(body: TradeIn):
    try:
        trade = add_trade(body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"trade": trade, "stats": stats()}


@app.post("/api/journal/{trade_id}/close")
def journal_close(trade_id: str, body: CloseIn):
    try:
        trade = close_trade(trade_id, body.model_dump())
    except KeyError as exc:
        raise HTTPException(404, "Trade not found") from exc
    return {"trade": trade, "stats": stats()}


@app.patch("/api/journal/{trade_id}")
def journal_patch(trade_id: str, body: TradePatch):
    try:
        trade = update_trade(trade_id, body.model_dump(exclude_none=True))
    except KeyError as exc:
        raise HTTPException(404, "Trade not found") from exc
    return {"trade": trade, "stats": stats()}


@app.delete("/api/journal/{trade_id}")
def journal_delete(trade_id: str):
    try:
        delete_trade(trade_id)
    except KeyError as exc:
        raise HTTPException(404, "Trade not found") from exc
    return {"ok": True, "stats": stats()}


@app.get("/api/history")
def history():
    return {"lots": broker_history()}
