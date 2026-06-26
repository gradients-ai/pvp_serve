"""End-to-end driver smoke test in random-agent mode (no GPU/model needed).

Plays every game to terminal, picking random legal human moves, and validates
the GameView contract + the bespoke structured shapes for othello & liars_dice.
Run: GOD_REPO_PATH=/path/to/G.O.D PVP_AGENT_KIND=random python -m tests.smoke
"""

import os
import random

os.environ.setdefault("PVP_AGENT_KIND", "random")

from app import session as driver  # noqa: E402
from app.contract import GameView  # noqa: E402

GAMES = ["liars_dice", "othello", "leduc_poker", "goofspiel", "gin_rummy"]


def _check_view(game: str, v: GameView) -> None:
    assert v.game == game
    assert isinstance(v.observation, str) and v.observation
    if v.toMove == "human":
        assert v.legalActions, f"{game}: human to move but no legal actions"
    if game == "othello" and v.structured is not None:
        s = v.structured
        assert len(s["board"]) == 64
        assert s["yourCount"] + s["agentCount"] == sum(1 for c in s["board"] if c)
        for flips in s["flipsByCell"].values():
            assert isinstance(flips, list)
    if game == "liars_dice" and v.structured is not None:
        s = v.structured
        assert s["dicePerPlayer"] == len(s["yourDice"])
        assert s["maxQuantity"] == s["totalDice"]


def play(game: str) -> str:
    rng = random.Random(game)
    ps = driver.create_session(game, human_seat_opt=0, model=None)
    s = driver.store.get(ps.sessionId)
    _check_view(game, ps.view)
    view = ps.view
    steps = 0
    while not view.isTerminal:
        steps += 1
        assert steps < 500, f"{game}: did not terminate"
        if view.toMove == "human":
            action = rng.choice(view.legalActions).id
            view = driver.apply_human_move(s, action)
        else:
            # agent/chance still pending — re-read (driver should have advanced already)
            view = driver.build_view(s)
            if view.toMove != "human" and not view.isTerminal:
                raise AssertionError(f"{game}: stuck with toMove={view.toMove}")
        _check_view(game, view)
    return f"{game}: OK ({steps} human steps, result={view.result}, returns={view.returns})"


if __name__ == "__main__":
    for g in GAMES:
        print(play(g))
    print("\nALL GAMES PASSED")
