import copy
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from baghchal import BaghChalGame
from training.mcts import MCTS
from training.self_play import select_action


class PerformanceEvaluator:
    def __init__(self, base_elo=1200):
        self.base_elo = base_elo
        self.elo_history = []
        self.game_results = []

    def calculate_elo_change(self, player_elo, opponent_elo, result, k_factor=32):
        expected_score = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
        return k_factor * (result - expected_score)

    def update_elo(self, current_elo, elo_change):
        return max(100, current_elo + elo_change)

    def _select_best_action(self, root_node):
        if not root_node.children:
            return None
        best_action = None
        best_visits = -1
        for action, child in root_node.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action
        return best_action

    def _mcts_action(self, mcts, game_state):
        root_node = mcts.search(game_state)
        return self._select_best_action(root_node)

    def _random_action(self, game_state):
        valid_actions = game_state.legal_actions()
        if not valid_actions:
            return None
        return valid_actions[np.random.randint(len(valid_actions))]

    def _heuristic_action(self, game_state):
        valid_actions = game_state.legal_actions()
        if not valid_actions:
            return None
        player = game_state.current_player()
        if player == "tiger":
            capture_actions = []
            for action in valid_actions:
                if len(action) != 2:
                    continue
                start, end = action
                if abs(end[0] - start[0]) == 2 or abs(end[1] - start[1]) == 2:
                    capture_actions.append(action)
            if capture_actions:
                return capture_actions[np.random.randint(len(capture_actions))]
        return valid_actions[np.random.randint(len(valid_actions))]

    def _build_agent(self, kind, model=None, num_simulations=100):
        if kind == "random":
            return {"kind": kind}
        if kind == "heuristic":
            return {"kind": kind}
        if kind == "model":
            if model is None:
                raise ValueError("model agent requires a model instance")
            return {
                "kind": kind,
                "mcts": MCTS(BaghChalGame(), model, num_simulations=num_simulations),
            }
        raise ValueError(f"Unknown agent kind: {kind}")

    def _choose_action(self, agent, game_state):
        if agent["kind"] == "random":
            return self._random_action(game_state)
        if agent["kind"] == "heuristic":
            return self._heuristic_action(game_state)
        if agent["kind"] == "model":
            return self._mcts_action(agent["mcts"], game_state)
        return None

    def _model_result_from_status(self, status, model_side):
        if status == "draw":
            return 0.5
        if status == "tiger_win":
            return 1.0 if model_side == "tiger" else 0.0
        if status == "goat_win":
            return 1.0 if model_side == "goat" else 0.0
        return 0.0

    def _play_single_game(self, tiger_agent, goat_agent, max_moves=200):
        game = BaghChalGame()
        move_count = 0

        while game.status == "ongoing" and move_count < max_moves:
            player = game.current_player()
            agent = tiger_agent if player == "tiger" else goat_agent
            action = self._choose_action(agent, game)
            if action is None or not game.apply_action(action):
                game.check_victory_conditions()
                if game.status == "ongoing":
                    game.status = "draw"
                break
            move_count += 1
            game.check_victory_conditions()

        if game.status == "ongoing":
            game.status = "draw"
        return game.status, move_count

    def _evaluate_model_matchup(
        self,
        model,
        opponent_kind,
        num_games=20,
        num_simulations=100,
        opponent_model=None,
        swap_sides=True,
    ):
        if opponent_kind == "model" and opponent_model is None:
            raise ValueError("opponent_model must be provided for opponent_kind='model'")

        model_points = 0.0
        tiger_side_games = 0
        goat_side_games = 0
        model_tiger_wins = 0
        model_goat_wins = 0
        draws = 0
        total_moves = 0

        for game_index in range(num_games):
            model_side = "tiger"
            if swap_sides and game_index % 2 == 1:
                model_side = "goat"

            model_agent = self._build_agent("model", model=model, num_simulations=num_simulations)
            if opponent_kind == "model":
                opponent_agent = self._build_agent("model", model=opponent_model, num_simulations=num_simulations)
            else:
                opponent_agent = self._build_agent(opponent_kind)

            tiger_agent = model_agent if model_side == "tiger" else opponent_agent
            goat_agent = model_agent if model_side == "goat" else opponent_agent

            status, moves = self._play_single_game(tiger_agent, goat_agent)
            total_moves += moves

            if model_side == "tiger":
                tiger_side_games += 1
                if status == "tiger_win":
                    model_tiger_wins += 1
            else:
                goat_side_games += 1
                if status == "goat_win":
                    model_goat_wins += 1

            if status == "draw":
                draws += 1

            model_points += self._model_result_from_status(status, model_side)

        max_points = float(num_games)
        model_score_rate = model_points / max_points if max_points > 0 else 0.0
        avg_game_length = total_moves / num_games if num_games > 0 else 0.0

        return {
            "opponent": opponent_kind if opponent_kind != "model" else "previous_checkpoint",
            "games": num_games,
            "model_points": model_points,
            "max_points": max_points,
            "model_score_rate": model_score_rate,
            "model_tiger_wins": model_tiger_wins,
            "model_goat_wins": model_goat_wins,
            "draws": draws,
            "tiger_side_games": tiger_side_games,
            "goat_side_games": goat_side_games,
            "avg_game_length": avg_game_length,
        }

    def evaluate_model_arena(
        self,
        model,
        num_games_per_opponent=20,
        num_simulations=100,
        include_random=True,
        include_heuristic=True,
        previous_model=None,
        swap_sides=True,
    ):
        opponents = []
        if include_random:
            opponents.append(("random", None))
        if include_heuristic:
            opponents.append(("heuristic", None))
        if previous_model is not None:
            opponents.append(("model", previous_model))

        matchup_results = []
        for opponent_kind, opponent_model in opponents:
            result = self._evaluate_model_matchup(
                model=model,
                opponent_kind=opponent_kind,
                opponent_model=opponent_model,
                num_games=num_games_per_opponent,
                num_simulations=num_simulations,
                swap_sides=swap_sides,
            )
            matchup_results.append(result)

        total_games = sum(item["games"] for item in matchup_results)
        total_points = sum(item["model_points"] for item in matchup_results)
        max_points = float(total_games)
        aggregate_score_rate = total_points / max_points if max_points > 0 else 0.0

        return {
            "total_games": total_games,
            "model_points": total_points,
            "max_points": max_points,
            "aggregate_score_rate": aggregate_score_rate,
            "matchups": matchup_results,
        }

    def evaluate_model_against_random(self, model, num_games=50, num_simulations=100):
        random_result = self._evaluate_model_matchup(
            model=model,
            opponent_kind="random",
            num_games=num_games,
            num_simulations=num_simulations,
            swap_sides=True,
        )

        score_rate = random_result["model_score_rate"]
        if score_rate <= 0 or score_rate >= 1:
            estimated_elo = self.base_elo
        else:
            elo_diff = 400 * np.log10(score_rate / (1 - score_rate))
            estimated_elo = self.base_elo + elo_diff

        return {
            "win_rate": score_rate,
            "loss_rate": max(0.0, 1.0 - score_rate),
            "draw_rate": random_result["draws"] / num_games if num_games else 0.0,
            "avg_game_length": random_result["avg_game_length"],
            "estimated_elo": estimated_elo,
            "total_games": num_games,
            "model_score_rate": score_rate,
            "model_points": random_result["model_points"],
            "max_points": random_result["max_points"],
            "model_tiger_wins": random_result["model_tiger_wins"],
            "model_goat_wins": random_result["model_goat_wins"],
        }

    def evaluate_model_against_baseline(self, model, baseline_model, num_games=20, num_simulations=100):
        result = self._evaluate_model_matchup(
            model=model,
            opponent_kind="model",
            opponent_model=baseline_model,
            num_games=num_games,
            num_simulations=num_simulations,
            swap_sides=True,
        )
        return {
            "model_wins": result["model_tiger_wins"] + result["model_goat_wins"],
            "baseline_wins": num_games - (result["model_tiger_wins"] + result["model_goat_wins"] + result["draws"]),
            "draws": result["draws"],
            "model_win_rate": result["model_score_rate"],
            "baseline_win_rate": max(0.0, 1.0 - result["model_score_rate"]),
            "draw_rate": result["draws"] / num_games if num_games else 0.0,
            "model_score_rate": result["model_score_rate"],
            "model_points": result["model_points"],
            "max_points": result["max_points"],
        }

    def analyze_game_quality(self, game_history):
        if not game_history:
            return {}

        game_lengths = []
        action_diversities = []
        value_consistencies = []

        for game_data in game_history:
            if len(game_data) >= 3:
                states, policies, values, masks = zip(*game_data)
                game_lengths.append(len(states))

                policy_entropies = []
                for policy in policies:
                    valid_probs = policy[policy > 0]
                    if len(valid_probs) > 1:
                        entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10))
                        policy_entropies.append(entropy)

                if policy_entropies:
                    action_diversities.append(np.mean(policy_entropies))

                if len(values) > 1:
                    value_changes = np.abs(np.diff(values))
                    value_consistencies.append(np.mean(value_changes))

        return {
            "avg_game_length": np.mean(game_lengths) if game_lengths else 0,
            "avg_action_diversity": np.mean(action_diversities) if action_diversities else 0,
            "avg_value_consistency": np.mean(value_consistencies) if value_consistencies else 0,
            "total_games": len(game_history),
            "total_positions": sum(len(game) for game in game_history),
        }

    def save_evaluation_results(self, results, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        with open(filename, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Evaluation results saved to {filename}")

    def plot_performance_trends(self, save_path="performance_trends.png"):
        if not self.elo_history:
            print("No ELO history to plot")
            return

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        iterations = range(len(self.elo_history))
        plt.plot(iterations, self.elo_history, "b-", linewidth=2)
        plt.title("ELO Rating Progression")
        plt.xlabel("Training Iteration")
        plt.ylabel("ELO Rating")
        plt.grid(True, alpha=0.3)

        if self.game_results:
            plt.subplot(2, 2, 2)
            win_rates = [result.get("win_rate", 0) for result in self.game_results]
            plt.plot(iterations[:len(win_rates)], win_rates, "g-", linewidth=2)
            plt.title("Win Rate Progression")
            plt.xlabel("Training Iteration")
            plt.ylabel("Win Rate")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Performance trends plot saved to {save_path}")


def create_elo_rating_system():
    return {
        "initial_rating": 1200,
        "k_factors": {
            "new_player": 40,
            "established": 32,
            "master": 16,
        },
        "rating_categories": {
            "beginner": (0, 1200),
            "intermediate": (1200, 1600),
            "advanced": (1600, 2000),
            "expert": (2000, 2400),
            "master": (2400, 3000),
        },
    }


def estimate_elo_from_win_rate(win_rate, opponent_elo=1200):
    if win_rate <= 0 or win_rate >= 1:
        return opponent_elo
    elo_diff = 400 * np.log10(win_rate / (1 - win_rate))
    return max(100, opponent_elo + elo_diff)
