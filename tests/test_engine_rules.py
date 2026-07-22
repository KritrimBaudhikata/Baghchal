import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baghchal import BaghChalGame


def test_turn_alternation_from_move_one():
    game = BaghChalGame()
    assert game.current_player() == "goat"
    assert game.place_goat((2, 2)) is True
    assert game.current_player() == "tiger"
    tiger_actions = game.legal_actions()
    assert tiger_actions
    assert game.apply_action(tiger_actions[0]) is True
    assert game.current_player() == "goat"


def test_side_to_move_enforcement():
    game = BaghChalGame()
    assert game.make_move((0, 0), (0, 1)) is False
    assert game.current_player() == "goat"


def test_illegal_move_rejection():
    game = BaghChalGame()
    assert game.place_goat((2, 2)) is True
    assert game.make_move((0, 0), (4, 4)) is False


def test_tiger_wins_at_five_captures():
    game = BaghChalGame()
    game.captured_goats = 5
    assert game.check_victory_conditions() == "tiger_win"


def test_threefold_repetition_draw_after_placement():
    game = BaghChalGame()
    game.goats_placed = 20
    game._update_phase()
    signature = game._state_signature(game.serialize_state_binary())
    game.state_history[signature] = 3
    assert game.check_victory_conditions() == "draw"


def test_serialization_roundtrip_preserves_turn_and_phase():
    game = BaghChalGame()
    game.place_goat((2, 2))
    tiger_action = game.legal_actions()[0]
    game.apply_action(tiger_action)
    state = game.serialize_state_binary()

    restored = BaghChalGame()
    restored.deserialize_state_binary(state)
    assert restored.current_player() == game.current_player()
    assert restored.goats_placed == game.goats_placed
    assert restored.captured_goats == game.captured_goats
    assert restored.phase == game.phase


def test_snapshot_roundtrip_preserves_state_history():
    game = BaghChalGame()
    game.place_goat((1, 1))
    before = dict(game.state_history)
    snap = game.snapshot()
    restored = BaghChalGame.from_snapshot(snap, adjacency_map=game.adjacency_map)
    assert restored.state_history == before
    assert restored.move_history == game.move_history
    assert restored.ply_count == game.ply_count


def test_tiger_capture_removes_goat_and_increments_count():
    game = BaghChalGame()
    game.board = [
        [1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
    ]
    game.goats_placed = 20
    game.phase = "movement"
    game.turn_player = "tiger"
    game.captured_goats = 0

    action = ((0, 0), (0, 2))
    assert action in game.legal_actions()
    assert game.apply_action(action) is True
    assert game.board[0][0] == 0
    assert game.board[0][1] == 0
    assert game.board[0][2] == 1
    assert game.captured_goats == 1


def test_adjacency_map_is_symmetric():
    game = BaghChalGame()
    for start, neighbors in game.adjacency_map.items():
        for end in neighbors:
            assert start in game.adjacency_map[end], f"missing reverse edge {end}->{start}"


def test_corner_has_orthogonal_and_diagonal_links():
    game = BaghChalGame()
    neighbors = set(game.adjacency_map[(0, 0)])
    assert (0, 1) in neighbors
    assert (1, 0) in neighbors
    assert (1, 1) in neighbors


def test_tiger_can_capture_vertically_from_corner():
    game = BaghChalGame()
    game.board = [
        [1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
    ]
    game.goats_placed = 20
    game.phase = "movement"
    game.turn_player = "tiger"
    game.captured_goats = 0

    action = ((0, 0), (2, 0))
    assert action in game.legal_actions()
    assert game.apply_action(action) is True
    assert game.board[1][0] == 0
    assert game.board[2][0] == 1
    assert game.captured_goats == 1


def test_goat_to_move_with_no_legal_moves_is_draw():
    game = BaghChalGame()
    # Corner goat blocked; tigers still have moves elsewhere.
    game.board = [
        [2, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ]
    game.goats_placed = 20
    game.phase = "movement"
    game.turn_player = "goat"
    game.captured_goats = 0
    assert game.legal_actions() == []
    assert game.check_victory_conditions() == "draw"
    assert game.status == "draw"
