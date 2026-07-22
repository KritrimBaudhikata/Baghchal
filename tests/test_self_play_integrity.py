import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baghchal import BaghChalGame
from game_actions import ACTION_SPACE_SIZE, action_to_index
from training.self_play import generate_self_play_data


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


def test_self_play_outputs_valid_training_samples():
    game = BaghChalGame()
    model = DummyModel()

    training_data, stats = generate_self_play_data(
        game=game,
        model=model,
        num_games=2,
        num_simulations=1,
        temperature=0.5,
        mcts_batch_size=1,
        return_stats=True,
    )

    assert stats["games_requested"] == 2
    assert stats["games_completed"] + stats["aborted_games"] == 2
    assert 0 <= stats["draw_rate"] <= 1
    assert len(training_data) <= stats["total_positions"]

    for state, policy, value, mask in training_data[:30]:
        assert state.shape == (5, 5, 5)
        assert policy.shape[0] == ACTION_SPACE_SIZE
        assert mask.shape[0] == ACTION_SPACE_SIZE
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert value in (-1.0, 0.0, 1.0)

        restored = BaghChalGame()
        restored.deserialize_state_binary(state)
        legal_indices = {action_to_index(action) for action in restored.legal_actions()}
        mask_indices = set(np.where(mask > 0.0)[0].tolist())
        assert mask_indices == legal_indices
