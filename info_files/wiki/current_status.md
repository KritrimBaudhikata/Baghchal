# Current Status — Baghchal

## Project goal

Build a correct Baghchal engine, AlphaZero-style self-play trainer, ELO evaluation system, and online playable version.

## Current known state

- Engine enforces explicit turn/phase state (`turn_player`, `phase`, `ply_count`) and alternation from move 1.
- Move application is centralized via `apply_action()` + `legal_actions()`.
- Action encoding is canonical and reversible (`ACTION_CATALOG`, `action_to_index`, `index_to_action`; `ACTION_SPACE_SIZE` ≈ 215).
- Full game snapshots (`snapshot()` / `restore()` / `clone()`) preserve `move_history` and `state_history` for MCTS repetition/draw detection.
- NN tensor (`serialize_state_binary`) remains the compact board+meta view for the network; MCTS nodes store full snapshots.
- MCTS uses efficient clone/snapshot (no adjacency deepcopy) and batched leaf evaluation.
- Training prefers `tf.data.Dataset` with prefetch; Sequence fallback remains.
- API: side selection, legal-moves endpoint, AI fallback + `/ai-move` retry so games cannot stick on AI turn after MCTS failures.
- Docker runs as non-root user `baghchal`.
- CI workflow: `.github/workflows/ci.yml` runs pytest on push/PR.

## Completed (this pass)

1. Snapshot/clone for full repetition history in MCTS (fewer deepcopies).
2. API AI failure recovery (`ai_pending`, fallback legal move, `POST /ai-move`).
3. `tf.data.Dataset` training path.
4. Docker non-root user.
5. GitHub Actions CI + version pins / `requirements-lock.txt`.
6. Tests for snapshot roundtrip and AI retry endpoint.
7. Capture application fix + symmetric adjacency generator (`ACTION_SPACE_SIZE` ≈ 215).
8. Goat-to-move with zero legal moves → `draw`.

## Test commands run

- `pytest -q tests/test_engine_rules.py` → **12 passed**
- Earlier full `pytest -q` → **30 passed, 1 skipped** (before goat-draw test)

## Next steps

1. Train a fresh model (prior checkpoints incompatible with ACTION_SPACE_SIZE 215).
2. Run API locally against the new checkpoint and smoke-test the UI.
3. Optional: rate-limit public API / smaller web-tier model.
4. Convert root GPU utility scripts to assert-based pytest style (warnings only).

## Latest fix

- Goat-to-move with zero legal moves → `draw` (avoids stuck `ongoing`; not a classical tiger win).

## Known risks/blockers

- Retrain required after action-space / adjacency fixes.
- Unauthenticated public API is a DoS risk without rate limits.
- GPU utility tests still emit return-value warnings (non-blocking).

## Recommended next mode

Standard — train baseline, then evaluate/deploy.
