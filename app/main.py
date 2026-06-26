"""FastAPI play server — the live backend for gradients-web /play.

Implements the GameConnector contract (src/lib/play/httpConnector.ts):
    POST  /session            { game, humanSeat, model? } -> PlaySession
    GET   /session/{id}                                   -> GameView
    POST  /session/{id}/move  { actionId }                -> GameView

All mutating calls run under a single process-wide lock, on the event loop's
main thread, because LLMBot's per-turn SIGALRM timeout is main-thread-only and
process-global. Run with a single uvicorn worker.
"""

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.contract import ChatRequest, CreateSessionRequest, GameView, MoveRequest, PlaySession
from app import chat as chat_proxy
from app import session as driver

app = FastAPI(title="pvp_serve", summary="Play the Gradients champion at OpenSpiel games")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serialises all game turns: SIGALRM (LLMBot's wall-clock) is process-global and
# main-thread-only, so only one turn may run at a time across the whole process.
_lock = asyncio.Lock()


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "agent_kind": settings.agent_kind, "model": settings.inference_model}


@app.post("/session", response_model=PlaySession)
async def create_session(req: CreateSessionRequest) -> PlaySession:
    async with _lock:
        try:
            return driver.create_session(req.game, req.humanSeat, req.model, req.playerId)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"unknown game {req.game!r}")


@app.get("/session/{session_id}", response_model=GameView)
async def get_state(session_id: str) -> GameView:
    async with _lock:
        s = driver.store.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
        return driver.build_view(s)


@app.post("/session/{session_id}/move", response_model=GameView)
async def submit_move(session_id: str, req: MoveRequest) -> GameView:
    async with _lock:
        s = driver.store.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id}")
        try:
            return driver.apply_human_move(s, req.actionId)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream a general chat reply from the champion as Server-Sent Events.

    Independent of the game harness — does not take the turn lock.
    """
    if not settings.is_llm:
        raise HTTPException(status_code=503, detail="chat needs a served model (PVP_AGENT_KIND=llm)")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    if len(req.messages) > settings.chat_max_messages:
        raise HTTPException(status_code=400, detail=f"too many messages (max {settings.chat_max_messages})")
    return StreamingResponse(
        chat_proxy.stream_chat(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
