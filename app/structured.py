"""Per-game `structured` builders for the rich Svelte renderers.

Only liars_dice (DiceTable.svelte) and othello (OthelloBoard.svelte) have
bespoke renderers and therefore need a structured payload — the shapes mirror
src/lib/play/structured.ts exactly. The other three games (leduc_poker,
goofspiel, gin_rummy) use the generic fallback renderer (raw observation text +
legal-action buttons), so they get structured=None.

All shapes are derived authoritatively from the pyspiel state, never guessed.
"""

from typing import Any

import pyspiel

# --- Othello -----------------------------------------------------------------

OTHELLO_SIZE = 8
OTHELLO_CELLS = OTHELLO_SIZE * OTHELLO_SIZE
OTHELLO_PASS_ACTION = OTHELLO_CELLS  # action 64 is "pass"

_BLACK = 1
_WHITE = 2


def othello_board(state: pyspiel.State) -> list[int]:
    """Parse the 8x8 board into a flat length-64 list: 0 empty, 1 black, 2 white.

    Cell index == pyspiel cell action id (row*8 + col), matching OthelloView.board.
    """
    board: list[int] = []
    for line in state.observation_string(0).splitlines():
        parts = line.split()
        # Grid rows look like:  "1 - - - o x - - - 1"  (rank, 8 cells, rank)
        if len(parts) >= 10 and parts[0].isdigit():
            for cell in parts[1:9]:
                board.append(0 if cell == "-" else _BLACK if cell == "x" else _WHITE)
    if len(board) != OTHELLO_CELLS:
        raise ValueError(f"othello board parse produced {len(board)} cells, expected {OTHELLO_CELLS}")
    return board


def _othello_flips(state: pyspiel.State, seat: int, action: int, before: list[int]) -> list[int]:
    """Cells that would flip to `seat`'s colour if it plays `action` (excludes the placed cell)."""
    nxt = state.clone()
    nxt.apply_action(action)
    after = othello_board(nxt)
    seat_disc = _BLACK if seat == 0 else _WHITE
    opp_disc = _WHITE if seat == 0 else _BLACK
    return [i for i in range(OTHELLO_CELLS) if before[i] == opp_disc and after[i] == seat_disc and i != action]


def othello_structured(state: pyspiel.State, human_seat: int, finished: bool) -> dict[str, Any]:
    board = othello_board(state)
    human_disc = _BLACK if human_seat == 0 else _WHITE
    agent_disc = _WHITE if human_seat == 0 else _BLACK
    black = board.count(_BLACK)
    white = board.count(_WHITE)

    human_to_move = (not finished) and (not state.is_terminal()) and state.current_player() == human_seat
    legal_cells = (
        [a for a in state.legal_actions(human_seat) if a < OTHELLO_PASS_ACTION] if human_to_move else []
    )
    flips_by_cell = {a: _othello_flips(state, human_seat, a, board) for a in legal_cells}

    return {
        "board": board,
        "size": OTHELLO_SIZE,
        "humanDisc": human_disc,
        "agentDisc": agent_disc,
        "black": black,
        "white": white,
        "yourCount": black if human_disc == _BLACK else white,
        "agentCount": black if agent_disc == _BLACK else white,
        "legalCells": legal_cells,
        "flipsByCell": flips_by_cell,
    }


def othello_action_meta(action: int) -> dict[str, Any]:
    if action == OTHELLO_PASS_ACTION:
        return {"kind": "pass"}
    return {"kind": "cell", "cell": action, "row": action // OTHELLO_SIZE, "col": action % OTHELLO_SIZE}


# --- Liar's Dice -------------------------------------------------------------

LIARS_DICE_CALL_ACTION = 60  # action 60 == call "Liar"; 0..59 are bids


def bid_from_action(action: int) -> tuple[int, int]:
    """Decode a liars_dice bid action into (quantity, face). Both 1-indexed."""
    return action // 6 + 1, action % 6 + 1


def parse_own_dice(state: pyspiel.State, seat: int) -> list[int]:
    """A player's own dice — the leading digit run of their information state."""
    info = state.information_state_string(seat)
    first = info.split()[0] if info.split() else ""
    return [int(c) for c in first if c.isdigit()]


def liars_dice_structured(
    state: pyspiel.State,
    human_seat: int,
    *,
    finished: bool,
    standing_bid: tuple[int, int] | None,
    challenge: dict[str, Any] | None,
) -> dict[str, Any]:
    agent_seat = 1 - human_seat
    your_dice = parse_own_dice(state, human_seat)
    dice_per_player = len(your_dice)
    total_dice = dice_per_player * state.num_players()
    showdown = finished or state.is_terminal()

    return {
        "yourDice": your_dice,
        "agentDiceCount": dice_per_player,
        "agentDice": parse_own_dice(state, agent_seat) if showdown else None,
        "dicePerPlayer": dice_per_player,
        "totalDice": total_dice,
        "bid": {"quantity": standing_bid[0], "face": standing_bid[1]} if standing_bid else None,
        "challenge": challenge,
        "maxQuantity": total_dice,
    }


def liars_dice_action_meta(action: int) -> dict[str, Any]:
    if action == LIARS_DICE_CALL_ACTION:
        return {"kind": "call"}
    quantity, face = bid_from_action(action)
    return {"kind": "bid", "quantity": quantity, "face": face}
