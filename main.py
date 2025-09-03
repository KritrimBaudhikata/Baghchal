from baghchal import BaghChalGame
from models.neural_network import create_bagh_chal_model
from training.training_loop import train_bagh_chal_model
from models.utils import PerformanceEvaluator
import os

def main():
    """Main training and evaluation function"""
    print("=== Baghchal AlphaZero Training System ===")
    
    # Initialize game and model
    print("Initializing game and model...")
    game = BaghChalGame()
    model = create_bagh_chal_model(action_space_size=65, learning_rate=0.001)
    
    # Create performance evaluator
    evaluator = PerformanceEvaluator(base_elo=1200)
    
    # Initial evaluation
    print("\n=== Initial Model Evaluation ===")
    initial_results = evaluator.evaluate_model_against_random(model, num_games=20, num_simulations=50)
    evaluator.save_evaluation_results(initial_results, "initial_evaluation.json")
    
    # Training
    print("\n=== Starting Training ===")
    trained_model, metrics = train_bagh_chal_model(
        game, model, 
        num_iterations=20, 
        games_per_iteration=20, 
        batch_size=64,
        learning_rate=0.001
    )
    
    # Final evaluation
    print("\n=== Final Model Evaluation ===")
    final_results = evaluator.evaluate_model_against_random(trained_model, num_games=50, num_simulations=100)
    evaluator.save_evaluation_results(final_results, "final_evaluation.json")
    
    # Performance analysis
    print("\n=== Performance Analysis ===")
    print(f"Initial ELO estimate: {initial_results['estimated_elo']:.0f}")
    print(f"Final ELO estimate: {final_results['estimated_elo']:.0f}")
    print(f"ELO improvement: {final_results['estimated_elo'] - initial_results['estimated_elo']:.0f}")
    
    # Save final model
    final_model_path = os.path.join("models", "final_bagh_chal_model.h5")
    trained_model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Generate performance plots
    evaluator.plot_performance_trends("final_performance_trends.png")
    
    print("\n=== Training Complete ===")
    print("Check the 'models' directory for saved models and evaluation results.")

if __name__ == "__main__":
    main()
