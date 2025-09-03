# File: self_play_debug.py
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from baghchal import BaghChalGame
from models.neural_network import create_bagh_chal_model
from training.self_play import generate_self_play_data

if __name__ == "__main__":
    # Initialize game and model
    game = BaghChalGame()
    model = create_bagh_chal_model(action_space_size=50)

    # Generate self-play data
    print("Generating self-play data...")
    training_data = generate_self_play_data(game, model, num_games=5)

    # Inspect the generated data
    print(f"Total data points: {len(training_data)}")
    for idx, (state, action, value) in enumerate(training_data[:10]):  # Print first 10 entries
        print(f"Data point {idx + 1}:")
        print(f"  State: {state}")
        print(f"  Action: {action}")
        print(f"  Value: {value}")
