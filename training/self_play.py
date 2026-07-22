import copy
import os
import sys
import numpy as np

from training.mcts import MCTS
from game_actions import ACTION_SPACE_SIZE, action_to_index

sys.path.append(os.path.abspath(".."))

DEFAULT_MCTS_BATCH_SIZE = 32
MAX_SELF_PLAY_MOVES = 200


def _build_policy_and_mask(root_node):
    valid_actions = list(root_node.children.keys())
    if not valid_actions:
        return None, None

    total_visits = sum(child.visit_count for child in root_node.children.values())
    if total_visits <= 0:
        return None, None

    action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for action, child in root_node.children.items():
        idx = action_to_index(action)
        action_mask[idx] = 1.0
        policy_target[idx] = child.visit_count / total_visits
    return policy_target, action_mask


def _final_value_from_status(status):
    if status == "tiger_win":
        return 1.0
    if status == "goat_win":
        return -1.0
    return 0.0


def _convert_history_to_training_data(history, final_value):
    training_data = []
    for experience in history:
        value = final_value if experience["player"] == "tiger" else -final_value
        training_data.append((
            experience["state"],
            experience["policy"],
            value,
            experience["mask"],
        ))
    return training_data


def _run_single_self_play_game(game, mcts, temperature, max_moves=MAX_SELF_PLAY_MOVES):
    history = []
    move_count = 0

    while game.status == "ongoing" and move_count < max_moves:
        state = game.serialize_state_binary()
        player = game.current_player()

        try:
            root_node = mcts.search(game)
        except Exception:
            return {
                "status": "aborted",
                "moves": move_count,
                "history": history,
                "reason": "mcts_failure",
            }

        policy_target, action_mask = _build_policy_and_mask(root_node)
        if policy_target is None or action_mask is None:
            game.check_victory_conditions()
            if game.status == "ongoing":
                game.status = "draw"
            break

        history.append({
            "state": copy.deepcopy(state),
            "policy": policy_target.copy(),
            "mask": action_mask.copy(),
            "player": player,
        })

        action = select_action(root_node, temperature)
        if action is None:
            return {
                "status": "aborted",
                "moves": move_count,
                "history": history,
                "reason": "no_selected_action",
            }

        if not game.apply_action(action):
            return {
                "status": "aborted",
                "moves": move_count,
                "history": history,
                "reason": "illegal_selected_action",
            }

        move_count += 1
        game.check_victory_conditions()

    if game.status == "ongoing":
        game.status = "draw"

    return {
        "status": game.status,
        "moves": move_count,
        "history": history,
        "reason": "completed",
    }


def _run_one_game_worker(args):
    model_path, num_simulations, temperature, seed, mcts_batch_size = args
    if seed is not None:
        np.random.seed(seed)

    from baghchal import BaghChalGame
    from models.neural_network import load_bagh_chal_model

    game = BaghChalGame()
    model = load_bagh_chal_model(model_path)
    mcts = MCTS(game, model, num_simulations=num_simulations, batch_size=mcts_batch_size or 1)

    result = _run_single_self_play_game(game, mcts, temperature, max_moves=MAX_SELF_PLAY_MOVES)
    if result["status"] in {"tiger_win", "goat_win", "draw"}:
        final_value = _final_value_from_status(result["status"])
        training_data = _convert_history_to_training_data(result["history"], final_value)
    else:
        training_data = []

    return {
        "training_data": training_data,
        "status": result["status"],
        "moves": result["moves"],
        "positions": len(result["history"]),
        "reason": result["reason"],
    }


def _empty_stats(num_games):
    return {
        "games_requested": num_games,
        "games_completed": 0,
        "tiger_wins": 0,
        "goat_wins": 0,
        "draws": 0,
        "aborted_games": 0,
        "total_moves": 0,
        "total_positions": 0,
        "avg_moves": 0.0,
        "tiger_win_rate": 0.0,
        "goat_win_rate": 0.0,
        "draw_rate": 0.0,
    }


def _finalize_stats(stats):
    completed = stats["games_completed"]
    if completed > 0:
        stats["avg_moves"] = stats["total_moves"] / completed
        stats["tiger_win_rate"] = stats["tiger_wins"] / completed
        stats["goat_win_rate"] = stats["goat_wins"] / completed
        stats["draw_rate"] = stats["draws"] / completed
    return stats


def generate_self_play_data(
    game,
    model,
    num_games=20,
    num_simulations=100,
    temperature=1.0,
    mcts_batch_size=None,
    return_stats=False,
):
    mcts = MCTS(
        game,
        model,
        num_simulations=num_simulations,
        batch_size=mcts_batch_size if mcts_batch_size is not None else DEFAULT_MCTS_BATCH_SIZE,
    )
    training_data = []
    stats = _empty_stats(num_games)

    for game_num in range(num_games):
        print(f"Generating game {game_num + 1}/{num_games}")
        game.reset()
        current_game = copy.deepcopy(game)

        result = _run_single_self_play_game(current_game, mcts, temperature, max_moves=MAX_SELF_PLAY_MOVES)
        status = result["status"]
        moves = result["moves"]
        positions = len(result["history"])

        stats["total_moves"] += moves
        stats["total_positions"] += positions

        if status in {"tiger_win", "goat_win", "draw"}:
            stats["games_completed"] += 1
            if status == "tiger_win":
                stats["tiger_wins"] += 1
            elif status == "goat_win":
                stats["goat_wins"] += 1
            else:
                stats["draws"] += 1
            final_value = _final_value_from_status(status)
            training_data.extend(_convert_history_to_training_data(result["history"], final_value))
        else:
            stats["aborted_games"] += 1

        print(f"Game {game_num + 1} completed. Result: {status}, Moves: {moves}")
        if game_num % 5 == 0:
            temperature = max(0.1, temperature * 0.95)

    stats = _finalize_stats(stats)
    print("\nSelf-play Statistics:")
    print(f"Total games requested: {stats['games_requested']}")
    print(f"Completed games: {stats['games_completed']}, Aborted: {stats['aborted_games']}")
    print(f"Tiger wins: {stats['tiger_wins']} ({stats['tiger_win_rate']*100:.1f}%)")
    print(f"Goat wins: {stats['goat_wins']} ({stats['goat_win_rate']*100:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draw_rate']*100:.1f}%)")
    print(f"Average moves per completed game: {stats['avg_moves']:.1f}")
    print(f"Total training samples: {len(training_data)}")

    if return_stats:
        return training_data, stats
    return training_data


def generate_self_play_data_parallel(
    game,
    model_path,
    num_games=20,
    num_simulations=100,
    temperature=1.0,
    num_workers=4,
    mcts_batch_size=None,
    return_stats=False,
):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import random

    if num_workers <= 1:
        from models.neural_network import load_bagh_chal_model
        model = load_bagh_chal_model(model_path)
        return generate_self_play_data(
            game,
            model,
            num_games,
            num_simulations,
            temperature,
            mcts_batch_size,
            return_stats=return_stats,
        )

    training_data = []
    stats = _empty_stats(num_games)
    mcts_batch_size = mcts_batch_size or DEFAULT_MCTS_BATCH_SIZE
    seeds = [random.randint(0, 2**31 - 1) for _ in range(num_games)]
    args_list = [
        (model_path, num_simulations, temperature, seeds[i], mcts_batch_size)
        for i in range(num_games)
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run_one_game_worker, args): idx for idx, args in enumerate(args_list)}
        for future in as_completed(futures):
            game_id = futures[future]
            try:
                result = future.result()
                training_data.extend(result["training_data"])
                stats["total_moves"] += result["moves"]
                stats["total_positions"] += result["positions"]
                if result["status"] in {"tiger_win", "goat_win", "draw"}:
                    stats["games_completed"] += 1
                    if result["status"] == "tiger_win":
                        stats["tiger_wins"] += 1
                    elif result["status"] == "goat_win":
                        stats["goat_wins"] += 1
                    else:
                        stats["draws"] += 1
                else:
                    stats["aborted_games"] += 1
                print(f"Game {game_id + 1}/{num_games} completed. Result: {result['status']}, Moves: {result['moves']}")
            except Exception as error:
                stats["aborted_games"] += 1
                print(f"Game {game_id + 1} failed: {error}")

    stats = _finalize_stats(stats)
    print("\nSelf-play Statistics (parallel):")
    print(f"Total games requested: {stats['games_requested']}")
    print(f"Completed games: {stats['games_completed']}, Aborted: {stats['aborted_games']}")
    print(f"Tiger wins: {stats['tiger_wins']} ({stats['tiger_win_rate']*100:.1f}%)")
    print(f"Goat wins: {stats['goat_wins']} ({stats['goat_win_rate']*100:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draw_rate']*100:.1f}%)")
    print(f"Average moves per completed game: {stats['avg_moves']:.1f}")
    print(f"Total training samples: {len(training_data)}")

    if return_stats:
        return training_data, stats
    return training_data


def select_action(node, temperature):
    actions = list(node.children.keys())
    if not actions:
        return None

    visits = np.array([node.children[action].visit_count for action in actions], dtype=np.float64)

    if temperature < 0.1:
        selected_idx = int(np.argmax(visits))
    else:
        visits = visits + np.random.random(len(visits)) * 0.1
        visits = visits ** (1 / max(temperature, 1e-6))
        probs = visits / visits.sum()
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()
        try:
            selected_idx = int(np.random.choice(len(probs), p=probs))
        except ValueError:
            selected_idx = int(np.random.randint(len(probs)))
    return actions[selected_idx]


def evaluate_model_performance(game, model, num_games=10, num_simulations=200):
    print("Evaluating model performance...")
    from models.utils import PerformanceEvaluator

    evaluator = PerformanceEvaluator()
    results = evaluator.evaluate_model_against_random(
        model=model,
        num_games=num_games,
        num_simulations=num_simulations,
    )
    print(
        f"Model score rate vs random (side-swapped): "
        f"{results['model_score_rate']:.3f} ({results['model_points']:.1f}/{results['max_points']:.1f})"
    )
    return results["model_score_rate"]
