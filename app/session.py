"""Game session lifecycle + the human-vs-champion turn driver.

A session owns one pyspiel game, the human's seat, and (in llm mode) one LLMBot
playing the other seat against the served model. The driver advances through
chance nodes and the agent's turns until it is the human's turn again or the
game is terminal — the same shape as evaluate_bots, but interactive: we step the
agent ourselves so the human can move in between.

IMPORTANT (concurrency): LLMBot.step() arms a per-turn SIGALRM wall-clock, and
signals only work on the main thread; SIGALRM is also process-global. So agent
turns must be serialised process-wide and run on the main thread. main.py holds
a single asyncio lock around every mutating call and runs the driver on the
event loop (not a threadpool) to satisfy this. Single uvicorn worker only.
"""

from __future__ import annotations

import functools
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.config import settings
from app.god_path import ensure_on_path

ensure_on_path()  # must run before importing core.*

from core.constants import ENVIRONMENT_CONFIGS, EnvironmentName  # noqa: E402
from core.models.pvp_models import ChatCompletionConfig, MemoryArea  # noqa: E402
from core.pvp import constants as pvp_cst  # noqa: E402
from core.pvp.bot import (  # noqa: E402
    ContextOverflowError,
    EmptyLegalActionsError,
    InvalidActionForfeitError,
    LLMBot,
    TurnTimeoutError,
    default_memories,
)
from core.pvp.chat import chat_completion, create_client  # noqa: E402
from core.pvp.game_eval import _AGENT_REGISTRY, config_id_for_seed  # noqa: E402
from core.pvp.memory import SlotMemory  # noqa: E402

from app import structured as sv  # noqa: E402
from app.contract import AgentMemory, AgentTurn, GameView, LegalAction, PlaySession  # noqa: E402

_FORFEIT_EXCEPTIONS = (TurnTimeoutError, ContextOverflowError, InvalidActionForfeitError)


# --- model wiring (built once) ----------------------------------------------


@functools.cache
def _chat_fn() -> tuple[Callable, ChatCompletionConfig]:
    config = ChatCompletionConfig(
        inference_model=settings.inference_model,
        base_url=settings.sglang_base_url,
        temperature=settings.temperature,
        seed=settings.seed,
        max_tokens=pvp_cst.PVP_TURN_MAX_TOKENS,
        read_timeout=float(pvp_cst.PVP_TURN_TIMEOUT_SECONDS) - 2.0,
        max_retries=1,
    )
    return functools.partial(chat_completion, create_client(config)), config


def _make_bot(game, agent_seat: int, agent, memories: dict[MemoryArea, SlotMemory]) -> LLMBot:
    chat_fn, config = _chat_fn()
    return LLMBot(game=game, player_id=agent_seat, chat_fn=chat_fn, config=config, agent=agent, memories=memories)


# --- session state -----------------------------------------------------------


@dataclass
class GameSession:
    id: str
    game_id: str
    human_seat: int
    agent_seat: int
    agent: Any
    game: Any
    state: Any
    rng: np.random.RandomState
    model: str
    player_id: str | None = None
    memories: dict[MemoryArea, SlotMemory] | None = None
    bot: LLMBot | None = None
    ply: int = 0
    last_agent_turn: AgentTurn | None = None
    finished: bool = False
    forfeit_returns: list[float] | None = None
    # liars_dice display tracking
    standing_bid: tuple[int, int] | None = None
    challenge: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    touched_at: float = field(default_factory=time.time)


class SessionStore:
    """In-memory session store. pyspiel states aren't serialisable, so single-process."""

    def __init__(self, ttl_seconds: int):
        self._sessions: dict[str, GameSession] = {}
        self._ttl = ttl_seconds

    def get(self, session_id: str) -> GameSession | None:
        self._evict()
        s = self._sessions.get(session_id)
        if s:
            s.touched_at = time.time()
        return s

    def put(self, s: GameSession) -> None:
        self._sessions[s.id] = s

    def _evict(self) -> None:
        cutoff = time.time() - self._ttl
        stale = [sid for sid, s in self._sessions.items() if s.touched_at < cutoff]
        for sid in stale:
            self._sessions.pop(sid, None)


store = SessionStore(settings.session_ttl_seconds)

_SESSION_SEED_RNG = np.random.RandomState()  # varies game variants across sessions

# Long-term memory persisted across a player's games, keyed by (playerId, game).
# In eval, long-term memory survives across the games of a matchup; here a
# returning player IS the opponent, so we carry the champion's notes between
# their games. In-memory (resets on server restart); good enough for now.
_LONG_TERM_STORE: dict[str, dict[int, str]] = {}


def _lt_key(player_id: str, game_id: str) -> str:
    return f"{player_id}:{game_id}"


def _load_long_term(session: GameSession, player_id: str | None) -> None:
    if not (player_id and session.memories):
        return
    saved = _LONG_TERM_STORE.get(_lt_key(player_id, session.game_id))
    if saved:
        session.memories[MemoryArea.LONG_TERM].slots.update(saved)


def _save_long_term(session: GameSession) -> None:
    if not (session.player_id and session.memories):
        return
    _LONG_TERM_STORE[_lt_key(session.player_id, session.game_id)] = dict(
        session.memories[MemoryArea.LONG_TERM].slots
    )


# --- driver ------------------------------------------------------------------


def create_session(game_id: str, human_seat_opt: int | str, model: str | None, player_id: str | None = None) -> PlaySession:
    env = EnvironmentName(game_id)
    agent_cls = _AGENT_REGISTRY[env]
    agent = agent_cls()

    seed = int(_SESSION_SEED_RNG.randint(0, 2**31 - 1))
    rng = np.random.RandomState(seed)
    config_id = config_id_for_seed(seed, ENVIRONMENT_CONFIGS[env])
    game = agent.load_game(agent.generate_params(config_id))
    state = game.new_initial_state()
    agent.setup_initial_state(state, seed)

    human_seat = (0 if rng.rand() < 0.5 else 1) if human_seat_opt == "random" else int(human_seat_opt)
    agent_seat = 1 - human_seat

    model_name = model or settings.inference_model
    session = GameSession(
        id=str(uuid.uuid4()),
        game_id=game_id,
        human_seat=human_seat,
        agent_seat=agent_seat,
        agent=agent,
        game=game,
        state=state,
        rng=rng,
        model=model_name if settings.is_llm else f"{model_name} (random)",
        player_id=player_id,
    )

    if settings.is_llm:
        session.memories = default_memories()
        session.bot = _make_bot(game, agent_seat, agent, session.memories)
        session.bot.restart_at(state)  # resets working memory; long-term loaded next
        _load_long_term(session, player_id)

    _advance(session)
    if session.finished or session.state.is_terminal():
        _on_game_end(session)
    store.put(session)
    return PlaySession(
        sessionId=session.id,
        game=session.game_id,  # type: ignore[arg-type]
        humanSeat=session.human_seat,  # type: ignore[arg-type]
        model=session.model,
        view=build_view(session),
    )


def apply_human_move(session: GameSession, action_id: int) -> GameView:
    state = session.state
    if session.finished or state.is_terminal() or state.current_player() != session.human_seat:
        return build_view(session)
    if action_id not in state.legal_actions(session.human_seat):
        raise ValueError(f"action {action_id} is not legal for the human right now")

    session.last_agent_turn = None
    _apply(session, session.human_seat, action_id)
    _advance(session)
    if session.finished or state.is_terminal():
        _on_game_end(session)
    return build_view(session)


def _advance(session: GameSession) -> None:
    """Advance through chance nodes and agent turns until the human moves or the game ends."""
    state = session.state
    while True:
        if session.finished or state.is_terminal():
            session.finished = True
            return
        if state.is_chance_node():
            _apply_chance(session)
            continue
        if state.current_player() == session.human_seat:
            return
        if not _agent_move(session):  # forfeit -> game over
            return


def _apply_chance(session: GameSession) -> None:
    outcomes = session.state.chance_outcomes()
    actions, probs = zip(*outcomes)
    chosen = int(session.rng.choice(actions, p=list(probs)))
    session.state.apply_action(chosen)


def _agent_move(session: GameSession) -> bool:
    """Play one agent turn. Returns False if the agent forfeited (game over)."""
    seat = session.agent_seat
    state = session.state
    try:
        if session.bot is not None:
            action = session.bot.step(state)
        else:
            legal = state.legal_actions(seat)
            action = int(session.rng.choice(legal))
    except _FORFEIT_EXCEPTIONS:
        _forfeit(session, forfeiting_seat=seat)
        return False
    except EmptyLegalActionsError:
        session.forfeit_returns = [0.0, 0.0]  # stuck -> draw
        session.finished = True
        return False

    label = _action_label(session, seat, action)
    session.last_agent_turn = AgentTurn(actionId=action, actionLabel=label)
    _apply(session, seat, action)
    return True


def _apply(session: GameSession, seat: int, action: int) -> None:
    session.state.apply_action(action)
    session.ply += 1
    if session.game_id == EnvironmentName.LIARS_DICE.value:
        _track_liars_dice(session, seat, action)


def _track_liars_dice(session: GameSession, mover_seat: int, action: int) -> None:
    if action == sv.LIARS_DICE_CALL_ACTION:
        if session.standing_bid is None:
            return
        qty, face = session.standing_bid
        all_dice = sv.parse_own_dice(session.state, 0) + sv.parse_own_dice(session.state, 1)
        actual = sum(d == face for d in all_dice)
        bidder = 1 - mover_seat  # caller challenges the previous mover's bid
        if session.state.is_terminal():
            bid_true = session.state.returns()[bidder] > 0
        else:
            bid_true = actual >= qty
        session.challenge = {
            "caller": mover_seat,
            "bid": {"quantity": qty, "face": face},
            "actualCount": actual,
            "bidWasTrue": bid_true,
        }
    else:
        session.standing_bid = sv.bid_from_action(action)
        session.challenge = None


def _forfeit(session: GameSession, forfeiting_seat: int) -> None:
    g = session.game
    returns = [g.max_utility()] * session.state.num_players()
    returns[forfeiting_seat] = g.min_utility()
    session.forfeit_returns = returns
    session.finished = True


def _on_game_end(session: GameSession) -> None:
    """At terminal: let the champion consolidate long-term notes, then persist them."""
    if session.bot is not None:
        _reflect(session)
    _save_long_term(session)


def _reflect(session: GameSession) -> None:
    """Best-effort long-term memory consolidation after a game (mirrors eval)."""
    if session.bot is None:
        return
    from core.models.pvp_models import GameOutcome

    rets = _returns(session)
    if rets is None:
        return
    margin = rets[session.agent_seat]
    outcome = GameOutcome.WIN if margin > 0 else GameOutcome.LOSS if margin < 0 else GameOutcome.DRAW
    try:
        session.bot.reflect(session.state, outcome)
    except Exception:
        pass


# --- view building -----------------------------------------------------------


def _returns(session: GameSession) -> list[float] | None:
    if session.forfeit_returns is not None:
        return session.forfeit_returns
    if session.state.is_terminal():
        return list(session.state.returns())
    return None


def _action_label(session: GameSession, seat: int, action: int) -> str:
    try:
        return session.state.action_to_string(seat, action)
    except (RuntimeError, AttributeError):
        return str(action)


def _legal_actions(session: GameSession) -> list[LegalAction]:
    state = session.state
    seat = session.human_seat
    out: list[LegalAction] = []
    for a in state.legal_actions(seat):
        out.append(LegalAction(id=a, label=_action_label(session, seat, a), meta=_action_meta(session.game_id, a)))
    return out


def _action_meta(game_id: str, action: int) -> dict[str, Any] | None:
    if game_id == EnvironmentName.LIARS_DICE.value:
        return sv.liars_dice_action_meta(action)
    if game_id == EnvironmentName.OTHELLO.value:
        return sv.othello_action_meta(action)
    return None


def _structured(session: GameSession) -> Any:
    if session.game_id == EnvironmentName.OTHELLO.value:
        return sv.othello_structured(session.state, session.human_seat, session.finished)
    if session.game_id == EnvironmentName.LIARS_DICE.value:
        return sv.liars_dice_structured(
            session.state,
            session.human_seat,
            finished=session.finished,
            standing_bid=session.standing_bid,
            challenge=session.challenge,
        )
    return None


def _agent_memory(session: GameSession) -> AgentMemory | None:
    if not session.memories:
        return None
    working = [v.strip() for v in session.memories[MemoryArea.WORKING].slots.values() if v.strip()]
    long_term = [v.strip() for v in session.memories[MemoryArea.LONG_TERM].slots.values() if v.strip()]
    return AgentMemory(working=working, longTerm=long_term)


def build_view(session: GameSession) -> GameView:
    state = session.state
    terminal = session.finished or state.is_terminal()
    returns = _returns(session)

    if terminal:
        to_move = None
    elif state.is_chance_node():
        to_move = "chance"
    elif state.current_player() == session.human_seat:
        to_move = "human"
    else:
        to_move = "agent"

    result = None
    if returns is not None:
        margin = returns[session.human_seat]
        result = "win" if margin > 0 else "loss" if margin < 0 else "draw"

    return GameView(
        game=session.game_id,  # type: ignore[arg-type]
        ply=session.ply,
        humanSeat=session.human_seat,  # type: ignore[arg-type]
        toMove=to_move,
        isTerminal=terminal,
        structured=_structured(session),
        observation=session.agent.format_state(state, session.human_seat),
        legalActions=_legal_actions(session) if to_move == "human" else [],
        lastAgentTurn=session.last_agent_turn,
        agentMemory=_agent_memory(session),
        result=result,
        returns=returns,
    )
