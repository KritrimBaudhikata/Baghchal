# State Map — Baghchal

## Game phases

### Goat placement
- Start: 4 tigers on corners; goats place on empty cells.
- Turn alternates from move 1 via `turn_player`.
- Phase ends when `goats_placed == 20` → `phase = "movement"`.

### Movement
- Both sides move on adjacency graph.
- Tigers capture by jump to empty landing; 5 captures → tiger win.
- No tiger moves → goat win.
- Goat to move with no moves → draw (unstick; not classical tiger win).
- Threefold on signatures after placement → draw (`state_history` counts).

## State representations

| Form | Contents | Used by |
|------|----------|---------|
| Live game | board, goats_placed/captured, turn_player, phase, status, ply_count, move_history, state_history | API session, self-play game object |
| `snapshot()` dict | Full mutable state including histories | MCTS node.state, clone/restore |
| `serialize_state_binary()` | `(5,5,5)` uint8: piece planes + turn/meta cells | NN input / training targets |

NN serialize does **not** carry full history; MCTS must use snapshots.

## Action flow

```text
state -> legal_actions() -> action_mask (via action_to_index)
      -> MCTS/NN -> selected_action -> apply_action() -> next_state
```

Same catalog for tests, MCTS, self-play, training, API, UI.

## API session shape
- `game`, `human_side`, `ai_side`, `ai_pending`, `ai_error`
- Endpoints: `POST /api/game`, `GET /api/game/{id}`, `GET .../legal-moves`, `POST .../move`, `POST .../ai-move`
