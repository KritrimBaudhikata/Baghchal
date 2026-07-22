# Context

## Completed Today
- Session audits: `/resume-session`, `/ponytail-audit`, `/code-review`, gitleaks (clean), `pip-audit` (clean); `safety` CLI broken in env.
- Fixed tiger capture: record `is_capture` before board mutation (`baghchal.py` `_apply_movement`).
- Fixed adjacency: generated symmetric Baghchal graph (orthogonals + X-diagonals); `ACTION_SPACE_SIZE` ≈ 215.
- Confirmed prior fixes already in tree: `snapshot`/`clone` for MCTS history; API AI fallback + `POST .../ai-move`.
- Goat-to-move with zero legal moves → `draw` (`check_victory_conditions`).
- Engine tests: capture, adjacency symmetry, corner links, vertical capture, goat-stalemate draw.
- Wiki: `current_status.md`, `state_map.md` updated; `/ponytail` left active for next session.

## Current Unfinished Item
- No trained checkpoint compatible with `ACTION_SPACE_SIZE` 215. Next: run training (`python main.py` / smoke config), then API UI smoke.

## Next Steps
- Train fresh model (`config.yaml` / `python main.py`); old `.keras` weights invalid.
- `pip install fastapi` if missing; `uvicorn api.app:app --reload --port 8080` smoke.
- Optional: top ponytail cuts (unused deps, dead `evaluate_state`, smoke scripts) before long train.
- Optional: public API rate limits / smaller web-tier net.
- Optional: assert-ify root GPU scripts (pytest warnings only).

## Known Risks / Blockers
- Retrain required after adjacency/action-space fix.
- Unauthenticated public API → DoS if cloud-exposed.
- `fastapi` missing in current conda env → API contract tests skip.
- Classical rules debate: goat immobilized is draw (not tiger win) by design.

## Checks
- `pytest -q tests/test_engine_rules.py` → 12 passed (includes goat-draw + capture/adjacency).
- Earlier full `pytest -q` → 30 passed, 1 skipped (before goat-draw test; expect +1).
- `gitleaks detect` → no leaks; `pip-audit -r requirements.txt` → no known vulns.
