# Baghchal Rework Roadmap

## Phase 0 — Audit and freeze assumptions

- Confirm exact Baghchal rules used by this implementation.
- Decide turn order during goat placement.
- Decide draw/repetition policy.
- Identify obsolete files and duplicate logic.

## Phase 1 — Rules engine stability

- Make state representation complete and testable.
- Fix legal move generation.
- Fix capture logic.
- Fix terminal conditions.
- Add rule tests.

## Phase 2 — Canonical action system

- Create single action list/encoder/decoder.
- Ensure no collisions.
- Generate legal action masks from game state.
- Use same mapping in MCTS, self-play, training, inference, and API.

## Phase 3 — MCTS and self-play

- Fix expansion/selection/backpropagation.
- Ensure value perspective is consistent.
- Generate valid policy targets.
- Prevent invalid/no-child expansion bugs.

## Phase 4 — Training

- Start with small model and fast smoke tests.
- Train against self-play only after game and MCTS tests pass.
- Save checkpoints and metrics.
- Avoid expensive training until pipeline is stable.

## Phase 5 — Evaluation

- Evaluate against random baseline.
- Add heuristic baseline.
- Add checkpoint-vs-checkpoint matches.
- Track ELO-like rating with limitations documented.

## Phase 6 — Online play

- Expose legal moves and AI move API.
- Build minimal board UI.
- Add human-vs-AI and AI-vs-AI modes.
- Ensure API never accepts illegal moves.

## Phase 7 — Cleanup and optimization

- Remove dead files.
- Remove print-debug noise.
- Add logging/debug flags.
- Optimize MCTS hot paths.
- Update README and wiki.
