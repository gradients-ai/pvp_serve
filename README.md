# pvp_serve

Human-vs-champion play server for the **Gradients (G.O.D)** PvP environment games.
It is the live backend behind the `/play` arena in `gradients-web`: a person plays
one of five OpenSpiel games against the trained tournament-champion LLM.

The authoritative game engine (pyspiel) and the agent (an `LLMBot` driving the
champion model via tool-calls) run here; the frontend only ever deals in opaque
action ids plus a rendered view of the state.

## What it serves

Implements the `GameConnector` contract from `gradients-web`
(`src/lib/play/httpConnector.ts` / `types.ts`):

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/session` | `{ game, humanSeat, model? }` | `PlaySession` |
| GET | `/session/{id}` | — | `GameView` |
| POST | `/session/{id}/move` | `{ actionId }` | `GameView` |
| GET | `/health` | — | status |

Games: `liars_dice`, `othello` (rich renderers — server emits a `structured`
view), and `leduc_poker`, `goofspiel`, `gin_rummy` (generic renderer — observation
text + legal-action buttons).

## How it relates to G.O.D

pvp_serve reuses G.O.D's `core/pvp` harness (the `LLMBot`, the per-game agents,
the game registry, the prompts) rather than reimplementing any game logic. G.O.D
is consumed as a **git submodule** at `./god`; only `core` is imported, and its
import surface is self-contained (no validator/docker/fiber/asyncpg deps), so the
heavy validator dependency tree is never installed.

Path resolution order (see `app/god_path.py`): `GOD_REPO_PATH` env → `./god`
submodule → sibling `../G.O.D` checkout (handy for local dev).

## Setup

```bash
# 1. dependencies (pyspiel/open_spiel installed separately — see note below)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. G.O.D source (production: submodule)
git submodule update --init --recursive       # pinned to G.O.D main
# (local dev alternative: skip the submodule and rely on a sibling ../G.O.D)

# 3. config
cp .env.example .env   # set PVP_SGLANG_BASE_URL, PVP_INFERENCE_MODEL, etc.
```

> **pyspiel / open_spiel** is not a `pip install .` dependency because wheels are
> platform-specific. Install it into the same environment separately (`pip install
> open_spiel` where wheels exist, otherwise build from source). `core.pvp` needs
> it at runtime.

## Run

Local dev with **no GPU/model** (agent plays random legal moves — exercises the
full session + rendering loop):

```bash
GOD_REPO_PATH=../G.O.D PVP_AGENT_KIND=random ./scripts/run.sh
```

Against the live champion (requires SGLang serving the model, see below):

```bash
# terminal 1 — serve the model (gated repo: export HF_TOKEN first)
export HF_TOKEN=hf_...
./scripts/serve_sglang.sh

# terminal 2 — the API
PVP_AGENT_KIND=llm PVP_SGLANG_BASE_URL=http://localhost:30000/v1 ./scripts/run.sh
```

## The champion model

As of 2026-06-26 the champion is the winner of environment tournament
`tourn_358aca49563e214e_20260622` (boss/final round):

- HF repo: `gradients-io-tournaments/tournament-tourn_358aca49563e214e_20260622-ac97eed9-69ff-4355-a012-2a9feaf3fd5f-5EEaxgnm`
- Base lineage: `Qwen/Qwen2.5-7B-Instruct` (continuous-train carry-forward)
- Tool-call parser: `qwen25`
- The repo is **gated** — needs an `HF_TOKEN` with access to `gradients-io-tournaments`.

To refresh after a new tournament, query the validator DB:
`SELECT winner_model_repo, winner_model_base FROM tournaments WHERE tournament_type='environment' AND status='completed' ORDER BY updated_at DESC LIMIT 1;`

## Concurrency note

`LLMBot.step()` arms a per-turn `SIGALRM` wall-clock; signals are main-thread-only
and `SIGALRM` is process-global. Therefore every game turn is serialised
process-wide (a single `asyncio` lock, turns run on the event loop) and the
server must run with **one uvicorn worker**. This caps throughput to one model
turn at a time — fine for launch; scaling past that means replacing the SIGALRM
timeout with a thread-pool-friendly one and sharding sessions across processes.

See `docs/NODE_SETUP.md` for deploying the model + server on a GPU box.
