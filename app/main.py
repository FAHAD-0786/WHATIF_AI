"""Web server. Listens on 127.0.0.1 only, so nothing else on your network can reach it."""
import asyncio
import secrets
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config, engine
from .crypto import Vault, WrongPassphrase
from .embed import backend, backend_error
from .llm import LLM
from .store import Store

vault = Vault(config.DATA_DIR)
store = Store(config.DATA_DIR / "whatif.db", vault)
answer_llm = LLM(config.LLM_MODE)
daily_llm = LLM(config.DAILY_MODE)
_session = {"token": None}
_fails = {"count": 0, "until": 0.0}


async def _nightly():
    """Every day at DAILY_UPDATE_AT, write missing daily insights (only while unlocked)."""
    hh, mm = map(int, config.DAILY_UPDATE_AT.split(":"))
    while True:
        now = datetime.now()
        nxt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        await asyncio.sleep((nxt - now).total_seconds())
        if vault.unlocked:
            try:
                await asyncio.to_thread(engine.catch_up, store, daily_llm)
            except Exception as e:
                print("nightly update failed:", e)


@asynccontextmanager
async def lifespan(_):
    task = asyncio.create_task(_nightly())
    yield
    task.cancel()


app = FastAPI(title="What If", docs_url=None, redoc_url=None, lifespan=lifespan)


def auth(x_token: str = Header(default="")):
    if not vault.unlocked or not _session["token"] or not secrets.compare_digest(x_token, _session["token"]):
        raise HTTPException(401, "Locked. Unlock with your passphrase.")


class Unlock(BaseModel):
    passphrase: str


class Entry(BaseModel):
    day: str | None = None
    text: str
    mood: str | None = None
    tags: list[str] = []


class Ask(BaseModel):
    question: str


def _day(d):
    try:
        return date.fromisoformat(d).isoformat() if d else date.today().isoformat()
    except ValueError:
        raise HTTPException(400, "Day must look like 2026-11-15")


@app.get("/")
def home():
    return FileResponse(config.ROOT / "app" / "static" / "index.html")


@app.get("/api/status")
def status():
    return {"initialised": vault.initialised, "unlocked": vault.unlocked, "embeddings": backend(),
            "answer_mode": answer_llm.mode, "daily_mode": daily_llm.mode}


@app.post("/api/unlock")
def unlock(b: Unlock):
    if time.time() < _fails["until"]:
        raise HTTPException(429, f"Too many wrong tries. Wait {int(_fails['until'] - time.time()) + 1} seconds.")
    try:
        vault.unlock(b.passphrase)
    except WrongPassphrase:
        _fails["count"] += 1
        if _fails["count"] >= 5:
            _fails.update(count=0, until=time.time() + 60)
        raise HTTPException(403, "Wrong passphrase.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    _fails["count"] = 0
    _session["token"] = secrets.token_urlsafe(32)
    return {"token": _session["token"]}


@app.post("/api/lock", dependencies=[Depends(auth)])
def lock():
    vault.lock()
    _session["token"] = None
    return {"ok": True}


@app.post("/api/entry", dependencies=[Depends(auth)])
def save_entry(e: Entry):
    day = _day(e.day)
    engine.save_entry(store, day, e.text, e.mood, e.tags)
    return {"saved": day}


@app.get("/api/entry/{day}", dependencies=[Depends(auth)])
def get_entry(day: str):
    return {"entry": store.get_entry(_day(day)), "insight": store.get_insight(_day(day))}


@app.get("/api/days", dependencies=[Depends(auth)])
def days():
    return {"days": store.days()}


@app.post("/api/daily/{day}", dependencies=[Depends(auth)])
def daily(day: str):
    try:
        r = engine.daily_update(store, daily_llm, _day(day))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if r is None:
        raise HTTPException(404, "No diary entry for that day.")
    return r


@app.post("/api/whatif", dependencies=[Depends(auth)])
def whatif(a: Ask):
    if not a.question.strip():
        raise HTTPException(400, "Ask something.")
    try:
        return engine.what_if(store, answer_llm, a.question)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/api/onthisday", dependencies=[Depends(auth)])
def onthisday():
    return {"items": engine.on_this_day(store)}


@app.get("/api/profile", dependencies=[Depends(auth)])
def profile():
    return {"profile": store.get_profile()}


@app.get("/api/history", dependencies=[Depends(auth)])
def history():
    return {"chats": store.chats()}


@app.get("/api/export", dependencies=[Depends(auth)])
def export():
    return store.export_all()


@app.delete("/api/wipe", dependencies=[Depends(auth)])
def wipe():
    store.wipe()
    return {"wiped": True}

@app.post("/api/reindex", dependencies=[Depends(auth)])
def reindex():
    if backend() != "fastembed":
        raise HTTPException(
            503,
            f"FastEmbed is not available: {backend_error()}"
        )

    count = engine.reindex_entries(store)

    return {
        "ok": True,
        "reindexed": count,
        "backend": backend(),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=config.PORT)
