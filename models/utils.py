import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from baghchal import BaghChalGame
from training.mcts import MCTS


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for Baghchal AI models.
    Includes ELO rating calculation, game analysis, and performance tracking.
    """
    
    def __init__(self, base_elo=1200):
        self.base_elo = base_elo
        self.elo_history = []
        self.game_results = []
        
    def calculate_elo_change(self, player_elo, opponent_elo, result, k_factor=32):
        """
        Calculate ELO rating change based on game result.
        
        :param player_elo: Current ELO rating of the player
        :param opponent_elo: Current ELO rating of the opponent
        :param result: 1 for win, 0.5 for draw, 0 for loss
        :param k_factor: K-factor for rating volatility
        :return: ELO change
        """
        expected_score = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
        elo_change = k_factor * (result - expected_score)
        return elo_change
    
    def update_elo(self, current_elo, elo_change):
        """Update ELO rating"""
        new_elo = current_elo + elo_change
        return max(100, new_elo)  # Minimum ELO of 100
    
    def evaluate_model_against_random(self, model, num_games=50, num_simulations=100):
        """
        Evaluate model performance against random play.
        
        :param model: Neural network model
        :param num_games: Number of games to play
        :param num_simulations: MCTS simulations per move
        :return: Win rate and ELO estimate
        """
        game = BaghChalGame()
        mcts = MCTS(game, model, num_simulations=num_simulations)
        
        wins = 0
        losses = 0
        draws = 0
        game_lengths = []
        
        print(f"Evaluating model against random play ({num_games} games)...")
        
        for game_num in range(num_games):
            game.reset()
            current_game = copy.deepcopy(game)
            move_count = 0
            
            while current_game.status == "ongoing" and move_count < 200:
                state = current_game.serialize_state_binary()
                player = current_game.current_player()
                
                if player == "tiger":
                    # Use MCTS for tiger
                    try:
                        root_node = mcts.search(state)
                        action = self._select_best_action(root_node)
                    except Exception as e:
                        print(f"MCTS failed in game {game_num}: {e}")
                        break
                else:
                    # Random play for goat
                    piece_type = 2
                    valid_actions = current_game.get_valid_moves(piece_type=piece_type)
                    if valid_actions:
                        action = valid_actions[np.random.randint(len(valid_actions))]
                    else:
                        break
                
                # Apply action
                try:
                    if len(action) == 1:
                        current_game.place_goat(action[0])
                    else:
                        current_game.make_move(action[0], action[1])
                except Exception as e:
                    print(f"Action failed in game {game_num}: {e}")
                    break
                
                move_count += 1
                
                if current_game.check_victory_conditions() != "ongoing":
                    break
            
            # Record result
            if current_game.status == "tiger_win":
                wins += 1
            elif current_game.status == "goat_win":
                losses += 1
            else:
                draws += 1
            
            game_lengths.append(move_count)
            
            if (game_num + 1) % 10 == 0:
                print(f"Completed {game_num + 1}/{num_games} games")
        
        win_rate = wins / num_games
        loss_rate = losses / num_games
        draw_rate = draws / num_games
        avg_game_length = np.mean(game_lengths)
        
        # Estimate ELO rating based on win rate against random (assumed ELO 1200)
        if win_rate > 0.5:
            # Calculate ELO difference needed for this win rate
            elo_diff = 400 * np.log10(win_rate / (1 - win_rate))
            estimated_elo = self.base_elo + elo_diff
        else:
            estimated_elo = self.base_elo
        
        print(f"\nEvaluation Results:")
        print(f"Games played: {num_games}")
        print(f"Wins: {wins} ({win_rate:.3f})")
        print(f"Losses: {losses} ({loss_rate:.3f})")
        print(f"Draws: {draws} ({draw_rate:.3f})")
        print(f"Average game length: {avg_game_length:.1f} moves")
        print(f"Estimated ELO rating: {estimated_elo:.0f}")
        
        return {
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'draw_rate': draw_rate,
            'avg_game_length': avg_game_length,
            'estimated_elo': estimated_elo,
            'total_games': num_games
        }
    
    def _select_best_action(self, root_node):
        """Select the best action from MCTS root node"""
        if not root_node.children:
            return None
        
        best_action = None
        best_visits = -1
        
        for action, child in root_node.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action
        
        return best_action
    
    def evaluate_model_against_baseline(self, model, baseline_model, num_games=20, num_simulations=100):
        """
        Evaluate model performance against a baseline model.
        
        :param model: Model to evaluate
        :param baseline_model: Baseline model for comparison
        :param num_games: Number of games to play
        :param num_simulations: MCTS simulations per move
        :return: Performance comparison
        """
        game = BaghChalGame()
        mcts_model = MCTS(game, model, num_simulations=num_simulations)
        mcts_baseline = MCTS(game, baseline_model, num_simulations=num_simulations)
        
        model_wins = 0
        baseline_wins = 0
        draws = 0
        
        print(f"Evaluating model against baseline ({num_games} games)...")
        
        for game_num in range(num_games):
            game.reset()
            current_game = copy.deepcopy(game)
            move_count = 0
            
            while current_game.status == "ongoing" and move_count < 200:
                state = current_game.serialize_state_binary()
                player = current_game.current_player()
                
                if player == "tiger":
                    # Model plays tiger
                    try:
                        root_node = mcts_model.search(state)
                        action = self._select_best_action(root_node)
                    except Exception as e:
                        print(f"Model MCTS failed: {e}")
                        break
                else:
                    # Baseline plays goat
                    try:
                        root_node = mcts_baseline.search(state)
                        action = self._select_best_action(root_node)
                    except Exception as e:
                        print(f"Baseline MCTS failed: {e}")
                        break
                
                # Apply action
                try:
                    if len(action) == 1:
                        current_game.place_goat(action[0])
                    else:
                        current_game.make_move(action[0], action[1])
                except Exception as e:
                    print(f"Action failed: {e}")
                    break
                
                move_count += 1
                
                if current_game.check_victory_conditions() != "ongoing":
                    break
            
            # Record result
            if current_game.status == "tiger_win":
                model_wins += 1
            elif current_game.status == "goat_win":
                baseline_wins += 1
            else:
                draws += 1
            
            if (game_num + 1) % 5 == 0:
                print(f"Completed {game_num + 1}/{num_games} games")
        
        model_win_rate = model_wins / num_games
        baseline_win_rate = baseline_wins / num_games
        draw_rate = draws / num_games
        
        print(f"\nModel vs Baseline Results:")
        print(f"Model wins: {model_wins} ({model_win_rate:.3f})")
        print(f"Baseline wins: {baseline_wins} ({baseline_win_rate:.3f})")
        print(f"Draws: {draws} ({draw_rate:.3f})")
        
        return {
            'model_wins': model_wins,
            'baseline_wins': baseline_wins,
            'draws': draws,
            'model_win_rate': model_win_rate,
            'baseline_win_rate': baseline_win_rate,
            'draw_rate': draw_rate
        }
    
    def analyze_game_quality(self, game_history):
        """
        Analyze the quality of games for training data.
        
        :param game_history: List of game states and actions
        :return: Quality metrics
        """
        if not game_history:
            return {}
        
        # Calculate various quality metrics
        game_lengths = []
        action_diversities = []
        value_consistencies = []
        
        for game_data in game_history:
            if len(game_data) >= 3:
                states, policies, values, masks = zip(*game_data)
                
                # Game length
                game_lengths.append(len(states))
                
                # Action diversity (entropy of policy)
                policy_entropies = []
                for policy in policies:
                    valid_probs = policy[policy > 0]
                    if len(valid_probs) > 1:
                        entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10))
                        policy_entropies.append(entropy)
                
                if policy_entropies:
                    action_diversities.append(np.mean(policy_entropies))
                
                # Value consistency (how much values change during the game)
                if len(values) > 1:
                    value_changes = np.abs(np.diff(values))
                    value_consistencies.append(np.mean(value_changes))
        
        quality_metrics = {
            'avg_game_length': np.mean(game_lengths) if game_lengths else 0,
            'avg_action_diversity': np.mean(action_diversities) if action_diversities else 0,
            'avg_value_consistency': np.mean(value_consistencies) if value_consistencies else 0,
            'total_games': len(game_history),
            'total_positions': sum(len(game) for game in game_history)
        }
        
        return quality_metrics
    
    def save_evaluation_results(self, results, filename=None):
        """Save evaluation results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {filename}")
    
    def plot_performance_trends(self, save_path="performance_trends.png"):
        """Plot performance trends over time"""
        if not self.elo_history:
            print("No ELO history to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # ELO progression
        plt.subplot(2, 2, 1)
        iterations = range(len(self.elo_history))
        plt.plot(iterations, self.elo_history, 'b-', linewidth=2)
        plt.title('ELO Rating Progression')
        plt.xlabel('Training Iteration')
        plt.ylabel('ELO Rating')
        plt.grid(True, alpha=0.3)
        
        # Win rate progression
        if self.game_results:
            plt.subplot(2, 2, 2)
            win_rates = [result.get('win_rate', 0) for result in self.game_results]
            plt.plot(iterations[:len(win_rates)], win_rates, 'g-', linewidth=2)
            plt.title('Win Rate Progression')
            plt.xlabel('Training Iteration')
            plt.ylabel('Win Rate')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance trends plot saved to {save_path}")


def create_elo_rating_system():
    """
    Create a standardized ELO rating system for Baghchal.
    
    :return: ELO rating system configuration
    """
    return {
        'initial_rating': 1200,
        'k_factors': {
            'new_player': 40,      # High volatility for new players
            'established': 32,      # Standard volatility
            'master': 16           # Low volatility for high-rated players
        },
        'rating_categories': {
            'beginner': (0, 1200),
            'intermediate': (1200, 1600),
            'advanced': (1600, 2000),
            'expert': (2000, 2400),
            'master': (2400, 3000)
        }
    }


def estimate_elo_from_win_rate(win_rate, opponent_elo=1200):
    """
    Estimate ELO rating from win rate against a known opponent.
    
    :param win_rate: Win rate (0.0 to 1.0)
    :param opponent_elo: ELO rating of the opponent
    :return: Estimated ELO rating
    """
    if win_rate <= 0 or win_rate >= 1:
        return opponent_elo
    
    # Calculate ELO difference needed for this win rate
    elo_diff = 400 * np.log10(win_rate / (1 - win_rate))
    estimated_elo = opponent_elo + elo_diff
    
    return max(100, estimated_elo)  # Minimum ELO of 100


# Import copy for the evaluator class
import copy
