"""Wire contract — 1:1 with gradients-web src/lib/play/types.ts.

Field names are camelCase to match the TypeScript client exactly, so responses
serialise straight onto the GameConnector interface with no transform.
"""

from typing import Any, Literal

from pydantic import BaseModel

GameId = Literal["liars_dice", "leduc_poker", "othello", "goofspiel", "gin_rummy"]
Seat = Literal[0, 1]
ToMove = Literal["human", "agent", "chance"] | None
GameResult = Literal["win", "loss", "draw"] | None


class LegalAction(BaseModel):
    id: int
    label: str
    meta: dict[str, Any] | None = None


class AgentTurn(BaseModel):
    actionId: int
    actionLabel: str
    thinking: str | None = None


class AgentMemory(BaseModel):
    working: list[str]
    longTerm: list[str]


class GameView(BaseModel):
    game: GameId
    ply: int
    humanSeat: Seat
    toMove: ToMove
    isTerminal: bool
    structured: Any
    observation: str
    legalActions: list[LegalAction]
    lastAgentTurn: AgentTurn | None = None
    agentMemory: AgentMemory | None = None
    result: GameResult = None
    returns: list[float] | None = None


class PlaySession(BaseModel):
    sessionId: str
    game: GameId
    humanSeat: Seat
    model: str
    view: GameView


class CreateSessionRequest(BaseModel):
    game: GameId
    humanSeat: Seat | Literal["random"] = "random"
    model: str | None = None


class MoveRequest(BaseModel):
    actionId: int


# --- general chat -----------------------------------------------------------


class ChatMessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageIn]
    # Optional per-request overrides; fall back to server defaults when unset.
    temperature: float | None = None
    maxTokens: int | None = None
