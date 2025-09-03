import numpy as np
import copy
import os
import sys
from training.mcts import MCTS

sys.path.append(os.path.abspath(".."))

def generate_self_play_data(game, model, num_games=20, num_simulations=100, temperature=1.0):
    """
    Enhanced self-play implementation with:
    - Valid action masking
    - Temperature-controlled exploration
    - Perspective-correct value targets
    - Repetition handling via MCTS
    - Better data quality and diversity
    """
    mcts = MCTS(game, model, num_simulations=num_simulations)
    training_data = []
    
    # Track statistics
    tiger_wins = 0
    goat_wins = 0
    draws = 0
    total_moves = 0

    for game_num in range(num_games):
        print(f"Generating game {game_num + 1}/{num_games}")
        game.reset()
        current_game = copy.deepcopy(game)
        history = []
        move_count = 0
        
        while current_game.status == "ongoing" and move_count < 200:  # Prevent infinite games
            # Get current state and player
            state = current_game.serialize_state_binary()
            player = current_game.current_player()
            
            # Run MCTS and get root node
            try:
                root_node = mcts.search(state)
            except Exception as e:
                print(f"MCTS search failed: {e}")
                break
            
            # Create action mask and policy targets
            valid_actions = list(root_node.children.keys())
            if not valid_actions:
                print("No valid actions found")
                break
                
            action_mask = np.zeros(65, dtype=np.float32)
            policy_target = np.zeros(65, dtype=np.float32)
            total_visits = sum(c.visit_count for c in root_node.children.values())
            
            for action, child in root_node.children.items():
                idx = action_to_index(action)
                if idx < 65:  # Ensure index is within bounds
                    action_mask[idx] = 1.0
                    policy_target[idx] = child.visit_count / total_visits
                
            # Store experience with mask
            history.append({
                'state': copy.deepcopy(state),
                'policy': policy_target,
                'mask': action_mask,
                'player': player
            })
            
            # Select and apply action
            action = select_action(root_node, temperature)
            
            try:
                if len(action) == 1:  # Goat placement
                    if not current_game.place_goat(action[0]):
                        print(f"Invalid goat placement at {action[0]}")
                        break
                else:  # Movement
                    if not current_game.make_move(action[0], action[1]):
                        print(f"Invalid move from {action[0]} to {action[1]}")
                        break
            except Exception as e:
                print(f"Action application failed: {e}")
                break
                
            move_count += 1
            
            # Check termination
            if current_game.check_victory_conditions() != "ongoing":
                final_value = 1.0 if current_game.status == "tiger_win" else -1.0
                
                # Update statistics
                if current_game.status == "tiger_win":
                    tiger_wins += 1
                elif current_game.status == "goat_win":
                    goat_wins += 1
                else:
                    draws += 1
                
                # Assign final values to all history positions with perspective correction
                for experience in history:
                    value = final_value if experience['player'] == "tiger" else -final_value
                    training_data.append((
                        experience['state'],
                        experience['policy'],
                        value,
                        experience['mask']
                    ))
                break
        
        total_moves += move_count
        print(f"Game {game_num+1} completed. Result: {current_game.status}, Moves: {move_count}")
        
        # Add some diversity by occasionally using different temperatures
        if game_num % 5 == 0:
            temperature = max(0.1, temperature * 0.95)
    
    # Print statistics
    print(f"\nSelf-play Statistics:")
    print(f"Total games: {num_games}")
    print(f"Tiger wins: {tiger_wins} ({tiger_wins/num_games*100:.1f}%)")
    print(f"Goat wins: {goat_wins} ({goat_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"Average moves per game: {total_moves/num_games:.1f}")
    print(f"Total training samples: {len(training_data)}")
        
    return training_data

def action_to_index(action):
    """
    Enhanced action mapping with better coverage of the action space.
    Maps game actions to network output indices (0-64).
    """
    if len(action) == 1:  # Goat placement
        row, col = action[0]
        return row * 5 + col
    else:  # Movement
        (from_row, from_col), (to_row, to_col) = action
        
        # Calculate movement direction and distance
        delta_row = to_row - from_row
        delta_col = to_col - from_col
        
        # Encode movement as direction + distance
        if abs(delta_row) == 2 or abs(delta_col) == 2:  # Jump move
            # Encode jump moves in indices 25-39
            direction = 0
            if delta_row == 2: direction = 0      # Down
            elif delta_row == -2: direction = 1   # Up
            elif delta_col == 2: direction = 2    # Right
            elif delta_col == -2: direction = 3   # Left
            
            base_idx = 25 + direction * 4
            return base_idx + (from_row * 5 + from_col) % 4
        else:
            # Encode regular moves in indices 40-64
            direction = 0
            if delta_row == 1: direction = 0      # Down
            elif delta_row == -1: direction = 1   # Up
            elif delta_col == 1: direction = 2    # Right
            elif delta_col == -1: direction = 3   # Left
            
            base_idx = 40 + direction * 6
            return base_idx + (from_row * 5 + from_col) % 6

def select_action(node, temperature):
    """
    Enhanced temperature-controlled action selection with exploration.
    """
    actions = list(node.children.keys())
    if not actions:
        return None
        
    visits = np.array([node.children[a].visit_count for a in actions])
    
    if temperature < 0.1 or node.parent is None:  # Greedy selection
        selected_idx = np.argmax(visits)
    else:  # Temperature sampling with exploration
        # Add small noise for exploration
        visits = visits + np.random.random(len(visits)) * 0.1
        visits = visits ** (1/temperature)
        probs = visits / visits.sum()
        
        # Ensure valid probabilities
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()
        
        try:
            selected_idx = np.random.choice(len(probs), p=probs)
        except ValueError:
            # Fallback to uniform random if probabilities are invalid
            selected_idx = np.random.randint(len(probs))
        
    return actions[selected_idx]

def evaluate_model_performance(game, model, num_games=10, num_simulations=200):
    """
    Evaluate model performance against random play.
    """
    print("Evaluating model performance...")
    
    mcts = MCTS(game, model, num_simulations=num_simulations)
    tiger_wins = 0
    goat_wins = 0
    
    for game_num in range(num_games):
        game.reset()
        current_game = copy.deepcopy(game)
        
        while current_game.status == "ongoing":
            state = current_game.serialize_state_binary()
            player = current_game.current_player()
            
            if player == "tiger":
                # Use MCTS for tiger
                root_node = mcts.search(state)
                action = select_action(root_node, temperature=0.1)  # Greedy
            else:
                # Random play for goat
                piece_type = 2
                valid_actions = current_game.get_valid_moves(piece_type=piece_type)
                if valid_actions:
                    action = valid_actions[np.random.randint(len(valid_actions))]
                else:
                    break
            
            # Apply action
            if len(action) == 1:
                current_game.place_goat(action[0])
            else:
                current_game.make_move(action[0], action[1])
            
            if current_game.check_victory_conditions() != "ongoing":
                if current_game.status == "tiger_win":
                    tiger_wins += 1
                elif current_game.status == "goat_win":
                    goat_wins += 1
                break
        
        print(f"Evaluation game {game_num+1}: {current_game.status}")
    
    win_rate = tiger_wins / num_games
    print(f"Tiger win rate against random: {win_rate:.3f} ({tiger_wins}/{num_games})")
    return win_rate