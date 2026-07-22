import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baghchal import BaghChalGame
from game_actions import ACTION_CATALOG, ACTION_SPACE_SIZE, action_to_index, index_to_action


def test_action_catalog_size_and_uniqueness():
    assert ACTION_SPACE_SIZE == len(ACTION_CATALOG)
    assert len(set(ACTION_CATALOG)) == ACTION_SPACE_SIZE
    indices = [action_to_index(action) for action in ACTION_CATALOG]
    assert len(set(indices)) == ACTION_SPACE_SIZE
    assert min(indices) == 0
    assert max(indices) == ACTION_SPACE_SIZE - 1


def test_action_bijection():
    for action in ACTION_CATALOG:
        idx = action_to_index(action)
        assert index_to_action(idx) == action


def test_placements_occupy_first_25_indices():
    for row in range(5):
        for col in range(5):
            action = ((row, col),)
            idx = action_to_index(action)
            assert idx == row * 5 + col


def test_all_legal_actions_are_encodable():
    game = BaghChalGame()
    for action in game.legal_actions():
        idx = action_to_index(action)
        assert 0 <= idx < ACTION_SPACE_SIZE
