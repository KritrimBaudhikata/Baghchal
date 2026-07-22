# AGENTS.md — Baghchal AI Project

This repository is for building a working Baghchal game engine, AlphaZero-style training pipeline, ELO evaluation system, and online playable interface.

## Primary Goal

Finish a reliable, testable Baghchal AI system that can:

1. Correctly enforce Baghchal rules.
2. Generate legal actions consistently.
3. Train through self-play using an AlphaZero-style policy/value network + MCTS.
4. Evaluate strength through repeatable baselines and ELO-style tracking.
5. Expose a playable online interface/API.
6. Keep documentation compact and useful.

## Working Principles for AI Agents

### 1. Do not start with broad repo scans

Before reading raw source files, read only:

1. `AGENTS.md`
2. `info_files/wiki/current_status.md`
3. `info_files/wiki/architecture.md`
4. `info_files/wiki/state_map.md`

Then wait for the current task unless explicitly told to continue.

### 2. Use context modes

#### Micro Mode
Use for small fixes.

- Read max 2–4 files.
- No broad plan.
- No repo scan.
- Implement directly.

#### Standard Mode
Use for normal feature work.

- Read max 5–10 relevant files.
- Make a short plan before edits.
- Use targeted file reads only.

#### Deep Mode
Use only for audits, architecture repair, training pipeline redesign, or hard bugs.

- Explain why Deep Mode is needed.
- Set a context budget before scanning.
- Prefer reading source in dependency order.

### 3. Documentation rules

- Keep `README.md` at repo root.
- Put all other markdown documentation under `info_files/`.
- Use `info_files/wiki/` for compact project handoff/state.
- Do not create random markdown files in the root.
- Update existing docs instead of creating new docs.
- Do not paste long code blocks into docs.
- Keep docs short enough to be useful for future agent restarts.

### 4. Source-code rules

- Do not refactor many files at once unless in Deep Mode.
- Do not optimize training before the game engine and action mapping are correct.
- Do not change public function signatures without updating tests and docs.
- Do not silently change Baghchal rules.
- Prefer pure functions for rules, action encoding, and state transitions.
- Add tests before or alongside rule/action/MCTS changes.
- Avoid hidden global state.
- Avoid print-heavy debugging in core modules; use logging or debug flags.

### 5. Current high-risk areas

Future agents should treat these as suspicious until verified:

- Turn handling during goat placement and movement phases.
- `serialize_state_binary()` / `deserialize_state_binary()` preserving full state.
- State repetition tracking using `set`, because sets are unordered.
- Action-space encoding collisions and fixed action size assumptions.
- MCTS expansion using action indices vs action IDs.
- Neural-network action mask implementation.
- Training data perspective: tiger/goat value targets.
- ELO evaluation against weak/random baselines.
- API/frontend coupling to incomplete game logic.

## Recommended work sequence

1. Stabilize rules engine.
2. Create canonical action encoder/decoder.
3. Fix state representation and cloning.
4. Fix MCTS using canonical actions.
5. Add unit tests for game rules, action mapping, MCTS expansion, and terminal states.
6. Build minimal self-play loop.
7. Train a small baseline model.
8. Build evaluation ladder: random, greedy heuristic, previous checkpoint, current checkpoint.
9. Add API and online playable UI.
10. Optimize performance and documentation.

## Done criteria

A task is not done unless:

- Relevant tests pass.
- The code can be run from a fresh terminal using documented commands.
- Any changed behavior is reflected in `info_files/wiki/current_status.md`.
- No unnecessary markdown files were created.
