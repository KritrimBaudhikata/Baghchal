import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import PerformanceEvaluator


class DummyModel:
    def predict(self, inputs, verbose=0):
        states, masks = inputs
        batch_size = states.shape[0]
        policy = np.array(masks, dtype=np.float32)
        totals = policy.sum(axis=1, keepdims=True)
        totals[totals == 0.0] = 1.0
        policy = policy / totals
        value = np.zeros((batch_size, 1), dtype=np.float32)
        return policy, value


def test_arena_reports_side_swapped_matchups():
    evaluator = PerformanceEvaluator(base_elo=1200)
    model = DummyModel()
    previous_model = DummyModel()

    report = evaluator.evaluate_model_arena(
        model=model,
        previous_model=previous_model,
        num_games_per_opponent=4,
        num_simulations=1,
        include_random=True,
        include_heuristic=True,
        swap_sides=True,
    )

    assert report["total_games"] == 12
    assert len(report["matchups"]) == 3
    assert 0.0 <= report["aggregate_score_rate"] <= 1.0

    for matchup in report["matchups"]:
        assert matchup["games"] == 4
        assert matchup["tiger_side_games"] == 2
        assert matchup["goat_side_games"] == 2
        assert 0.0 <= matchup["model_score_rate"] <= 1.0


def test_against_random_returns_model_score_rate():
    evaluator = PerformanceEvaluator(base_elo=1200)
    model = DummyModel()
    result = evaluator.evaluate_model_against_random(model, num_games=4, num_simulations=1)

    assert "model_score_rate" in result
    assert "estimated_elo" in result
    assert 0.0 <= result["model_score_rate"] <= 1.0
    assert result["total_games"] == 4
