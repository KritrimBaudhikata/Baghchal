# Architecture — Baghchal

## Intended architecture

```text
BaghchalGame / Rules Engine
        ↓
Canonical State + Action Encoder (game_actions)
        ↓
MCTS Search (snapshot states + batched NN eval)
        ↓
Self-Play Data Generation
        ↓
Policy/Value Neural Network Training (tf.data)
        ↓
Evaluation + ELO Ladder (models/utils)
        ↓
API + Online Play UI (api/)
```

## Current folder structure summary

```text
api/                 FastAPI + static UI
models/              NN create/load + PerformanceEvaluator
training/            MCTS, self-play, training_loop
tests/               pytest suite
info_files/wiki/     Agent handoff / status
baghchal.py          Core engine (apply_action, snapshot, serialize)
game_actions.py      ACTION_CATALOG / action_to_index / index_to_action
config.yaml          Training/eval defaults
main.py              Train + eval entry
run_inference.py     Checkpoint smoke games
Dockerfile           Non-root uvicorn serve
.github/workflows/   CI pytest
```

## Core responsibilities

### Game engine (`baghchal.py`)
- Board, turn/phase, legal moves, apply_action, terminals.
- `serialize_state_binary()` → NN tensor only.
- `snapshot()` / `restore()` / `clone()` → full state including `state_history` for MCTS.

### Action system (`game_actions.py`)
- Canonical catalog (~215), bijection index↔action, shared by MCTS/self-play/NN/API.

### MCTS (`training/mcts.py`)
- Selection / expansion / eval / backprop; node.state is a **snapshot dict**.
- Batched leaf `model.predict`; sequential path if `batch_size <= 1`.

### Neural network (`models/neural_network.py`)
- Dual-head policy/value; masked softmax; `.keras` load/save.
- Action space size must match `ACTION_SPACE_SIZE`.

### Training (`training/training_loop.py`, `self_play.py`)
- Self-play → replay buffer → prefer `tf.data` fit; checkpoint `.keras` + manifest.
- Optional parallel workers via saved `_current_iter.keras`.

### API (`api/app.py`)
- In-memory games; human/AI sides; `/legal-moves`; AI fallback + `/ai-move`.
- Rules always via engine; model only chooses AI moves.

## Architecture warnings
- Retrain after any change to `ACTION_CATALOG` / adjacency.
- Do not use NN tensor alone inside MCTS if draw-by-repetition matters.
- Do not let UI bypass `/legal-moves` or `apply_action`.
- Random-baseline ELO is not true strength.
