#!/usr/bin/env python3
"""
Load a saved Baghchal model and run inference: one full game (AI vs random or AI vs AI).
Use this to validate a checkpoint before using it in the web API.
"""
import argparse
import os
import sys

import numpy as np

from baghchal import BaghChalGame
from game_actions import action_to_index
from models.neural_network import load_bagh_chal_model
from training.mcts import MCTS
from training.self_play import select_action


def run_one_game(game, model, num_simulations=50, tiger_ai=True, goat_ai=False, max_moves=200):
    """
    Run one game. If tiger_ai: tiger uses MCTS+model; else random.
    If goat_ai: goat uses MCTS+model; else random.
    Returns final status and move count.
    """
    game.reset()
    mcts = MCTS(game, model, num_simulations=num_simulations)
    move_count = 0

    while game.status == "ongoing" and move_count < max_moves:
        state = game.serialize_state_binary()
        player = game.current_player()
        piece_type = 1 if player == "tiger" else 2
        use_ai = (player == "tiger" and tiger_ai) or (player == "goat" and goat_ai)

        if use_ai:
            try:
                root = mcts.search(game)
                action = select_action(root, temperature=0.1)
            except Exception as e:
                print(f"MCTS failed: {e}", file=sys.stderr)
                break
        else:
            valid = game.legal_actions()
            if not valid:
                break
            action = valid[np.random.randint(len(valid))]

        if action is None:
            break

        if not game.apply_action(action):
            break
        move_count += 1
        game.check_victory_conditions()

    return game.status, move_count


def main():
    parser = argparse.ArgumentParser(description="Run inference with a saved Baghchal model")
    parser.add_argument("model_path", type=str, help="Path to .keras or .h5 model")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--games", type=int, default=1, help="Number of games to run")
    parser.add_argument("--goat-ai", action="store_true", help="Use AI for goat too (else random)")
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"Error: Model file not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.model_path}...")
    model = load_bagh_chal_model(args.model_path)
    game = BaghChalGame()

    results = {"tiger_win": 0, "goat_win": 0, "draw": 0}
    for i in range(args.games):
        status, moves = run_one_game(
            game, model,
            num_simulations=args.simulations,
            tiger_ai=True,
            goat_ai=args.goat_ai,
        )
        if status == "tiger_win":
            results["tiger_win"] += 1
        elif status == "goat_win":
            results["goat_win"] += 1
        else:
            results["draw"] += 1
        print(f"Game {i+1}: {status} ({moves} moves)")

    print(f"\nResults: Tiger {results['tiger_win']} | Goat {results['goat_win']} | Draw {results['draw']}")
    print("Inference run completed successfully.")


if __name__ == "__main__":
    main()
