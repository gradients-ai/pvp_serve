"""Per-game `structured` builders for the rich Svelte renderers.

Only liars_dice (DiceTable.svelte) and othello (OthelloBoard.svelte) have
bespoke renderers and therefore need a structured payload — the shapes mirror
src/lib/play/structured.ts exactly. The other three games (leduc_poker,
goofspiel, gin_rummy) use the generic fallback renderer (raw observation text +
legal-action buttons), so they get structured=None.

All shapes are derived authoritatively from the pyspiel state, never guessed.
"""

import re
from typing import Any

import pyspiel

# --- shared card helpers -----------------------------------------------------

_SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}  # ♠♥♦♣


def card_token_to_str(tok: str) -> str:
    """'As' -> 'A♠', 'Td' -> '10♦'. Pass through anything unexpected."""
    if len(tok) < 2 or tok[-1] not in _SUIT_SYMBOL:
        return tok
    rank = tok[:-1]
    rank = "10" if rank == "T" else rank
    return f"{rank}{_SUIT_SYMBOL[tok[-1]]}"

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


# --- Leduc Poker -------------------------------------------------------------

_LEDUC_RANKS = ["J", "Q", "K", "A"]


def _leduc_card(card_id: int) -> str:
    rank_idx, suit_idx = card_id // 2, card_id % 2
    rank = _LEDUC_RANKS[rank_idx] if rank_idx < len(_LEDUC_RANKS) else f"?{card_id}"
    return f"{rank}{'♠' if suit_idx == 0 else '♥'}"


def _find(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def leduc_poker_structured(state: pyspiel.State, human_seat: int) -> dict[str, Any]:
    info = state.information_state_string(human_seat)
    private = _find(r"\[Private: (-?\d+)\]", info)
    public = _find(r"\[Public: (-?\d+)\]", info)
    rnd = _find(r"\[Round (\d+)\]", info)
    pot = _find(r"\[Pot: (\d+)\]", info)
    money = _find(r"\[Money: ([\d ]+)\]", info)

    private_id = int(private) if private and private != "-10000" else None
    public_id = int(public) if public and public != "-10000" else None
    your_card = _leduc_card(private_id) if private_id is not None else None
    public_card = _leduc_card(public_id) if public_id is not None else None
    is_pair = private_id is not None and public_id is not None and private_id // 2 == public_id // 2

    chips = (money or "").split()
    your_chips = int(chips[human_seat]) if len(chips) > human_seat else None
    opp_chips = int(chips[1 - human_seat]) if len(chips) > (1 - human_seat) else None

    return {
        "yourCard": your_card,
        "publicCard": public_card,
        "isPair": is_pair,
        "round": int(rnd) if rnd else 1,
        "pot": int(pot) if pot else 0,
        "yourChips": your_chips,
        "oppChips": opp_chips,
    }


def leduc_action_meta(label: str) -> dict[str, Any]:
    return {"kind": label.strip().lower()}  # fold / call / check / raise


# --- Goofspiel ---------------------------------------------------------------


def _line_after(text: str, key: str) -> str:
    for line in text.splitlines():
        if line.startswith(key):
            return line[len(key):].strip()
    return ""


def goofspiel_structured(state: pyspiel.State, human_seat: int) -> dict[str, Any]:
    info = state.information_state_string(human_seat)
    your_hand = [int(x) for x in _line_after(info, "P0 hand:" if human_seat == 0 else "P1 hand:").split()]
    your_bids = [int(x) for x in _line_after(info, f"P{human_seat} action sequence:").split() if x.isdigit()]
    revealed = [int(x) for x in _line_after(info, "Point card sequence:").split()]
    points = [int(x) for x in _line_after(info, "Points:").split()]

    total = len(your_hand) + len(your_bids)  # your bids + remaining hand == 1..N
    current_prize = revealed[-1] if revealed else None
    upcoming = sorted(set(range(1, total + 1)) - set(revealed)) if total else []

    return {
        "currentPrize": current_prize,
        "yourHand": sorted(your_hand),
        "upcomingPrizes": upcoming,
        "pastPrizes": revealed[:-1] if revealed else [],
        "yourScore": points[human_seat] if len(points) > human_seat else 0,
        "oppScore": points[1 - human_seat] if len(points) > (1 - human_seat) else 0,
        "totalPrizes": total,
    }


def goofspiel_action_meta(state: pyspiel.State, seat: int, action: int) -> dict[str, Any]:
    label = state.action_to_string(seat, action)
    card = _find(r"Bid:\s*(\d+)", label)
    return {"kind": "bid", "card": int(card) if card else None}


# --- Gin Rummy ---------------------------------------------------------------

_CARD_TOKEN = re.compile(r"\b([A2-9TJQK]|10)([shdc])\b")


def _gin_cards_from_grid(obs: str, owner: str) -> list[str]:
    """Parse the player's hand from the ASCII grid that follows 'Player0:'/'Player1:'."""
    lines = obs.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        if line.strip().startswith(owner):
            capturing = True
            continue
        if capturing:
            if line.strip().startswith("+--") or not line.strip():
                if out:  # stop at the closing border once we've collected cards
                    if line.strip().startswith("+--"):
                        break
                continue
            for rank, suit in _CARD_TOKEN.findall(line):
                out.append(card_token_to_str(f"{rank}{suit}"))
    return out


def gin_rummy_structured(state: pyspiel.State, human_seat: int) -> dict[str, Any]:
    obs = state.observation_string(human_seat)
    owner = f"Player{human_seat}"
    hand = _gin_cards_from_grid(obs, owner)
    deadwood = _find(rf"{owner}: Deadwood=(\d+)", obs) or _find(r"Deadwood=(\d+)", obs)
    upcard = _find(r"Upcard:\s*([A-Z0-9T]+[shdc])", obs)
    stock = _find(r"Stock size:\s*(\d+)", obs)
    phase = _find(r"Phase:\s*(\w+)", obs)
    discard_line = _line_after(obs, "Discard pile:")
    discard_tokens = _CARD_TOKEN.findall(discard_line)
    discard_top = card_token_to_str("".join(discard_tokens[-1])) if discard_tokens else None

    return {
        "yourHand": hand,
        "deadwood": int(deadwood) if deadwood else None,
        "upcard": card_token_to_str(upcard) if upcard else None,
        "discardTop": discard_top,
        "stockSize": int(stock) if stock else None,
        "phase": phase,
    }


_CARD_FULL = re.compile(r"^([A2-9TJQK]|10)([shdc])$")


def gin_action_meta(label: str) -> dict[str, Any]:
    low = label.lower()
    if "draw upcard" in low:
        return {"kind": "draw_upcard"}
    if "draw stock" in low:
        return {"kind": "draw_stock"}
    if "pass" in low:
        return {"kind": "pass"}
    if "knock" in low:
        return {"kind": "knock"}
    if "gin" in low:
        return {"kind": "gin"}
    # Discards are labelled "Player: N Action: <card>" (bare card token, no verb).
    m = re.search(r"Action:\s*(\S+)", label)
    if m and _CARD_FULL.match(m.group(1)):
        return {"kind": "discard", "card": card_token_to_str(m.group(1))}
    return {"kind": "other"}
