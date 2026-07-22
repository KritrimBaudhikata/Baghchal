#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baghchal import BaghChalGame
from game_actions import ACTION_SPACE_SIZE
from models.neural_network import create_bagh_chal_model
from models.utils import PerformanceEvaluator


def test_basic_functionality():
    game = BaghChalGame()
    state = game.serialize_state_binary()
    game.deserialize_state_binary(state)
    tiger_moves = game.get_valid_moves(1)
    goat_moves = game.get_valid_moves(2)
    assert state.shape == (5, 5, 5)
    assert isinstance(tiger_moves, list)
    assert isinstance(goat_moves, list)


def test_neural_network():
    model = create_bagh_chal_model(action_space_size=ACTION_SPACE_SIZE, learning_rate=0.001)
    test_state = np.random.random((1, 5, 5, 5))
    test_mask = np.ones((1, ACTION_SPACE_SIZE))
    policy, value = model.predict([test_state, test_mask], verbose=0)
    assert policy.shape == (1, ACTION_SPACE_SIZE)
    assert value.shape == (1, 1)


def test_performance_evaluator():
    evaluator = PerformanceEvaluator(base_elo=1200)
    elo_change = evaluator.calculate_elo_change(1200, 1200, 1.0)
    new_elo = evaluator.update_elo(1200, elo_change)
    assert new_elo > 1200


def test_game_mechanics():
    game = BaghChalGame()
    assert game.place_goat((2, 2)) is True
    assert len(game.get_valid_moves(1)) > 0
    assert game.check_victory_conditions() in {"ongoing", "tiger_win", "goat_win", "draw"}
