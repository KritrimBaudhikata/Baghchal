# First Audit Prompt — Baghchal Rework

Act as a senior AI game-engine architect and AlphaZero implementation reviewer.

Use Deep Mode for this one-time audit.

## Goal

Study the current Baghchal codebase enough to create a safe rework plan. Do not implement fixes yet.

## Read order

Read only these files first:

1. `baghchal.py`
2. `game_actions.py`
3. `training/mcts.py`
4. `training/self_play.py`
5. `training/training_loop.py`
6. `models/neural_network.py`
7. `models/utils.py`
8. `tests/`
9. `api/app.py`
10. `README.md`

If a file does not exist, note it and continue.

## Audit areas

1. Baghchal rule correctness.
2. Turn/phase handling.
3. Legal move generation.
4. Capture logic.
5. Repetition/draw logic.
6. State serialization/deserialization.
7. Action-space encoding/decoding.
8. MCTS correctness.
9. Self-play data quality.
10. Neural-network input/output/masking.
11. Training loop stability.
12. Evaluation/ELO validity.
13. API/playable UI readiness.
14. Dead/unwanted/inefficient code.
15. Test coverage gaps.

## Output format

Return:

1. Top 10 critical issues, ranked.
2. Minimum fix sequence.
3. Files that should be changed first.
4. Tests that should be added first.
5. What should not be touched yet.
6. Recommended roadmap: Phase 0, Phase 1, Phase 2, Phase 3.

Do not implement code in this audit response.
