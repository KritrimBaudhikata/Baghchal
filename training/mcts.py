import numpy as np
import math
import sys
import os
import copy
from baghchal import BaghChalGame # Changed import to use the updated class

# Add parent directory to path
sys.path.append(os.path.abspath(".."))

## Create a class to represent each node in the search tree.
class MCTSNode:
    def __init__(self, state, valid_actions, parent=None, action=None):
        self.state = state  # The game state at this node
        self.parent = parent  # Parent node
        self.action = action  # Action taken to reach this node
        self.children = {}  # Map: action -> child node
        self.visit_count = 0  # N(s)
        self.total_value = 0  # W(s)
        self.prior = 0  # P(s, a)
        self.valid_actions = valid_actions #Track all possible actions
        self._value = 0 # new: cached value estimate
    
    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

    def is_fully_expanded(self):
        return len(self.children) == len(self.valid_actions)  # Fixed expansion check


## The algorithm includes the main components: Selection, Expansion, Simulation, and Backpropagation.
class MCTS:
    def __init__(self, game, model, c_puct=1.25, num_simulations=200):
        self.base_game = game  # BaghChalGame instance
        self.model = model  # Neural network for policy and value
        self.c_puct = c_puct  # Exploration parameter
        self.num_simulations = num_simulations  # Number of simulations per move

    def search(self, root_state):
        """Improved search with temporary game states"""
        # CREATE TEMPORARY GAME FOR THIS SEARCH
        temp_game = copy.deepcopy(self.base_game)
        temp_game.deserialize_state_binary(root_state)
        
        # Get initial valid actions
        current_player = temp_game.current_player()
        piece_type = 1 if current_player == "tiger" else 2
        valid_actions = temp_game.get_valid_moves(piece_type=piece_type)
        root = MCTSNode(root_state, valid_actions)  # Create the root node

        for _ in range(self.num_simulations):
            node = root
            temp_game = copy.deepcopy(self.base_game) # Fresh copy of the game for each simulation
            
            # Selection
            while node.is_fully_expanded() and node.children:
                action, node = self._select(node)
                temp_game.simulate_action(node.state, action)  # Update the game state
            
            # Expansion
            if temp_game.status == "ongoing" and not node.is_fully_expanded():
                self._expand(node, temp_game)

            # Simulation
            value = self._simulate(temp_game)

            # Backpropagation
            self._backpropagate(node, value, temp_game.current_player())

        return root

    def _select_child(self, node):
        """
        Select the child node with the highest UCB score.
        Improved UCB calculation with prior scaling
        """
        total_n = math.sqrt(sum(c.visit_count for c in node.children.values()))
        best_score = -np.inf
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            # Improved exploration formula with prior scaling
            exploration = self.c_puct * child.prior * total_n / (1 + child.visit_count)
            score = child.q_value + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _expand(self, node, temp_game):
        """
        Expand the node by adding a new child for each valid action.
        Safer expansion with action masking
        """
        temp_game = BaghChalGame()  # Create a temporary game instance
        temp_game.deserialize_state_binary(node.state)  # Load the state into the temporary game

        # Get valid actions from temp game state
        current_player = temp_game.current_player()
        piece_type = 1 if current_player == "tiger" else 2
        valid_actions = temp_game.get_valid_moves(piece_type=piece_type)
        
        # Create action mask for the neural network
        action_mask = np.zeros(65) # match the size of the output layer
        for idx, action in enumerate(valid_actions):
            action_mask[idx] = 1
        
        # Get neural network predictions
        state_input = temp_game.serialize_state_binary()[np.newaxis, ...]
        policy, value = self.model.predict([state_input, action_mask[np.newaxis, ...]], verbose =0)  # Mask all actions initially
        
        # Check victory conditions before expansion
        if temp_game.check_victory_conditions() != "ongoing":
            print(f"Game ended with status: {temp_game.status}. No expansion possible.", flush=True)
            return

        # Add child nodes for each valid move
        for action_idx, action in enumerate(valid_actions):
            # SIMULATE ACTION ON TEMP GAME COPY
            sim_game = copy.deepcopy(temp_game)
            new_state = sim_game.simulate_action(node.state, action)
            
            # GET VALID ACTIONS FOR NEW STATE
            new_player = sim_game.current_player()
            new_piece_type = 1 if new_player == "tiger" else 2
            new_valid_actions = sim_game.get_valid_moves(new_piece_type)
            
            child = MCTSNode(
                state=new_state,
                valid_actions=new_valid_actions,
                parent=node,
                action=action
            )
            child.prior = policy[0][action_idx]
            child._value = value[0][0]
            node.children[action] = child
    
    def _state_to_input(self, game):
        """
        Convert the game state to the input format expected by the neural network.
        :param game: BaghChalGame instance
        :return: numpy array of shape (1,3,5,5)
        """
        # Convert game state to input tensor
        board = np.array(game.board)
        tiger_layer = (board == 1).astype(float)
        goat_layer = (board == 2).astype(float)
        empty_layer = (board == 0).astype(float)
        
        # Combine layers into input tensor and add batch dimension
        input_tensor = np.stack([tiger_layer, goat_layer, empty_layer], axis=0)
        return np.expand_dims(input_tensor, axis=0)   # Shape: (1,3,5,5)
    
    def _simulate(self, temp_game):
        """
        Use the neural network to evaluate the node's state.
        Direct value prediction without manual penalties
        """
        if temp_game.status != "ongoing":
            return 1 if temp_game.status == "tiger_win" else -1
        
        # USE NETWORK PREDICTION DIRECTLY
        state_input = temp_game.serialize_state_binary()[np.newaxis, ...]
        action_mask = np.zeros(65)  # DUMMY MASK FOR SIMULATION
        _, value = self.model.predict([state_input, action_mask[np.newaxis, ...]], verbose=0)
        return value[0][0]

    def _backpropagate(self, node, value, player):
        """
        Update visit counts and total values along the path back to the root.
        Perspective aware backpropagation
        """
        while node:
            node.visit_count += 1
            # insert value for alternating turns
            node.total_value += value if player == "tiger" else -value
            node = node.parent
            player = "tiger" if player == "goat" else "goat"