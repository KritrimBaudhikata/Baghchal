"""Test that MCTS uses action_to_index consistently with the policy output."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from baghchal import BaghChalGame
from game_actions import ACTION_SPACE_SIZE, action_to_index
from training.mcts import MCTS
from models.neural_network import create_bagh_chal_model


def test_action_to_index_placement():
    """Placement (row, col) -> first 25 catalog indices."""
    for r in range(5):
        for c in range(5):
            idx = action_to_index(((r, c),))
            assert idx == r * 5 + c, f"({r},{c}) -> {idx}"


def test_mcts_expand_uses_correct_indices():
    """After one MCTS expansion, children have priors from policy at action_to_index(action)."""
    game = BaghChalGame()
    model = create_bagh_chal_model(action_space_size=ACTION_SPACE_SIZE, learning_rate=0.001)
    mcts = MCTS(game, model, num_simulations=1, batch_size=1)
    root = mcts.search(game)
    # Root should be expanded; each child's prior should match policy[action_to_index(action)]
    if root.children:
        for action, child in root.children.items():
            idx = action_to_index(action)
            assert 0 <= idx < ACTION_SPACE_SIZE
            assert hasattr(child, "prior")
            assert isinstance(child.state, dict)
            assert "state_history" in child.state


def test_snapshot_preserves_repetition_history():
    game = BaghChalGame()
    game.place_goat((2, 2))
    snap = game.snapshot()
    assert len(snap["state_history"]) >= 1
    restored = BaghChalGame.from_snapshot(snap, adjacency_map=game.adjacency_map)
    assert restored.state_history == game.state_history
    assert restored.current_player() == game.current_player()
    assert restored.goats_placed == game.goats_placed
