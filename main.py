"""Baghchal AlphaZero training entry point. Supports config file and --resume checkpoint."""
import argparse
import os

from baghchal import BaghChalGame
from game_actions import ACTION_SPACE_SIZE
from models.neural_network import create_bagh_chal_model, load_bagh_chal_model
from training.training_loop import train_bagh_chal_model
from models.utils import PerformanceEvaluator


def load_training_config():
    try:
        from config_loader import load_config, get
        load_config()
        return {
            "num_iterations": get("training.num_iterations", 20),
            "games_per_iteration": get("training.games_per_iteration", 20),
            "batch_size": get("training.batch_size", 64),
            "learning_rate": get("training.learning_rate", 0.001),
            "num_simulations": get("training.num_simulations", 100),
            "save_dir": get("training.save_dir", "models"),
            "mcts_batch_size": get("training.mcts_batch_size", 32),
            "parallel_workers": get("training.parallel_workers", 0),
            "initial_games": get("evaluation.initial_games", 20),
            "initial_simulations": get("evaluation.initial_simulations", 50),
            "final_games": get("evaluation.final_games", 50),
            "final_simulations": get("evaluation.final_simulations", 100),
            "base_elo": get("evaluation.base_elo", 1200),
        }
    except Exception:
        return {
            "num_iterations": 20,
            "games_per_iteration": 20,
            "batch_size": 64,
            "learning_rate": 0.001,
            "num_simulations": 100,
            "save_dir": "models",
            "initial_games": 20,
            "initial_simulations": 50,
            "final_games": 50,
            "final_simulations": 100,
            "base_elo": 1200,
            "mcts_batch_size": 32,
            "parallel_workers": 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Baghchal AlphaZero Training")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint model (.keras or .h5) to resume training",
    )
    args = parser.parse_args()

    cfg = load_training_config()
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print("=== Baghchal AlphaZero Training System ===")

    # Initialize game and model (new or from checkpoint)
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}...")
        model = load_bagh_chal_model(args.resume)
        # Try to read iteration from manifest next to checkpoint
        import json
        manifest_path = os.path.join(os.path.dirname(args.resume), "checkpoint_manifest.json")
        initial_iteration = 0
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    initial_iteration = manifest.get("iteration", 0)
                print(f"Resuming from iteration {initial_iteration}")
            except Exception:
                pass
    else:
        if args.resume:
            print(f"Resume path not found: {args.resume}, starting from scratch.")
        print("Initializing new model...")
        model = create_bagh_chal_model(action_space_size=ACTION_SPACE_SIZE, learning_rate=cfg["learning_rate"])
        initial_iteration = 0

    game = BaghChalGame()
    evaluator = PerformanceEvaluator(base_elo=cfg["base_elo"])

    # Initial evaluation (skip if resuming past iteration 0)
    if initial_iteration == 0:
        print("\n=== Initial Model Evaluation ===")
        initial_results = evaluator.evaluate_model_against_random(
            model, num_games=cfg["initial_games"], num_simulations=cfg["initial_simulations"]
        )
        evaluator.save_evaluation_results(initial_results, "initial_evaluation.json")
    else:
        initial_results = {"estimated_elo": cfg["base_elo"]}

    # Training
    print("\n=== Starting Training ===")
    trained_model, metrics = train_bagh_chal_model(
        game,
        model,
        num_iterations=cfg["num_iterations"],
        games_per_iteration=cfg["games_per_iteration"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        save_dir=save_dir,
        initial_iteration=initial_iteration,
        num_simulations=cfg["num_simulations"],
        mcts_batch_size=cfg.get("mcts_batch_size", 32),
        parallel_workers=cfg.get("parallel_workers", 0),
    )

    # Final evaluation
    print("\n=== Final Model Evaluation ===")
    final_results = evaluator.evaluate_model_against_random(
        trained_model, num_games=cfg["final_games"], num_simulations=cfg["final_simulations"]
    )
    evaluator.save_evaluation_results(final_results, "final_evaluation.json")

    # Performance analysis
    print("\n=== Performance Analysis ===")
    print(f"Initial ELO estimate: {initial_results['estimated_elo']:.0f}")
    print(f"Final ELO estimate: {final_results['estimated_elo']:.0f}")
    print(f"ELO improvement: {final_results['estimated_elo'] - initial_results['estimated_elo']:.0f}")

    # Save final model (.keras)
    default_model = os.path.join(save_dir, "final_bagh_chal_model.keras")
    trained_model.save(default_model)
    print(f"\nFinal model saved to: {default_model}")

    evaluator.plot_performance_trends("final_performance_trends.png")
    print("\n=== Training Complete ===")
    print("Check the 'models' directory for saved models and evaluation results.")


if __name__ == "__main__":
    main()
