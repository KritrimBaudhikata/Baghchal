import copy
import numpy as np

class BaghChalGame:
    def __init__(self):
        """
        Initialize the Bagh Chal board.
        """
        # 5x5 board: 0 = empty, 1 = tiger, 2 = goat
        self.board = [
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
        ]
        self.goats_placed = 0  # Number of goats placed on the board
        self.captured_goats = 0  # Number of goats captured by tigers
        # Define adjacency map for valid moves (based on the Bagh Chal graph)
        self.adjacency_map = self._generate_bagh_chal_adjacency_map()
        self.status = "ongoing" # Game status: ongoing, tiger_wins, goat_wins
        self.move_history = []  # Track the history of moves made
        self.state_history = set()  # Track the history of game states in set class
    
    ## Function to check if the state is repetitive   
    def is_repetitive_state(self):
        current_state = self.serialize_state_binary()
        
        #Skip repetitive state check for the initial state
        if len(self.state_history) == 0:
            print(f"Skipping repetitive state check for the initial state: {current_state}")
            self.state_history.add(current_state)
            return False
        
        # Check recent history for repetitive states
        recent_history = list(self.state_history)[-20:]
        if current_state in recent_history:
            print(f"State {current_state} is repetitive within recent history.")
            return True
        # if len(self.move_history) > 0:  # Only add state after at least one move
        self.state_history.add(current_state)
        return False

    ## Function to generate the adjacency map for the Bagh Chal board
    def _generate_bagh_chal_adjacency_map(self):
        """
        Generates the complete adjacency map for the Bagh Chal board.
        """
        bagh_chal_adjacency_map = {
            (0, 0): [(0, 1), (1, 1)],  # Top-left corner
            (0, 1): [(0, 0), (0, 2), (1, 1)],
            (0, 2): [(0, 1), (0, 3), (1, 2)],
            (0, 3): [(0, 2), (0, 4), (1, 3)],
            (0, 4): [(0, 3), (1, 3)],  # Top-right corner

            (1, 0): [(0, 0), (1, 1), (2, 0)],
            (1, 1): [(0, 1), (1, 0), (1, 2), (2, 1), (0, 0), (0, 2)],
            (1, 2): [(0, 2), (1, 1), (1, 3), (2, 2)],
            (1, 3): [(0, 3), (1, 2), (1, 4), (2, 3), (0, 4), (0, 2)],
            (1, 4): [(0, 4), (1, 3), (2, 4)],

            (2, 0): [(1, 0), (2, 1), (3, 0)],
            (2, 1): [(1, 1), (2, 0), (2, 2), (3, 1), (1, 0), (1, 2)],
            (2, 2): [(1, 2), (2, 1), (2, 3), (3, 2), (1, 1), (1, 3)],
            (2, 3): [(1, 3), (2, 2), (2, 4), (3, 3), (1, 2), (1, 4)],
            (2, 4): [(1, 4), (2, 3), (3, 4)],

            (3, 0): [(2, 0), (3, 1), (4, 0)],
            (3, 1): [(2, 1), (3, 0), (3, 2), (4, 1), (2, 0), (2, 2)],
            (3, 2): [(2, 2), (3, 1), (3, 3), (4, 2), (2, 1), (2, 3)],
            (3, 3): [(2, 3), (3, 2), (3, 4), (4, 3), (2, 2), (2, 4)],
            (3, 4): [(2, 4), (3, 3), (4, 4)],

            (4, 0): [(3, 0), (4, 1), (3, 1)],
            (4, 1): [(3, 1), (4, 0), (4, 2), (3, 0), (3, 2)],
            (4, 2): [(3, 2), (4, 1), (4, 3), (3, 1), (3, 3)],
            (4, 3): [(3, 3), (4, 2), (4, 4), (3, 2), (3, 4)],
            (4, 4): [(3, 4), (4, 3), (3, 3)],
        }
        return bagh_chal_adjacency_map
    
    ## Function to track repeated moves
    def is_repetitive_move(self, piece_type, start_pos=None, end_pos=None):
        """
        Check if the move repeats a back-and-forth cycle more than twice.
        
        :param piece_type: The type of piece making the move ('tiger' or 'goat').
        :param start_pos: The starting position of the piece (optional).
        :param end_pos: The ending position of the piece (optional).
        :return: True if the move repeats a cycle, False otherwise.
        """
        # Ensure there are at least 4 moves in the history to check
        if len(self.move_history) < 4:
            return False

        # Filter moves for the same piece type
        relevant_moves = [move for move in self.move_history if move[0] == piece_type]

        # Check if specific positions are provided
        if start_pos and end_pos:
            # Check for back-and-forth pattern: A → B → A → B
            if (
                relevant_moves[-4:] == [
                    (piece_type, start_pos, end_pos),
                    (piece_type, end_pos, start_pos),
                    (piece_type, start_pos, end_pos),
                    (piece_type, end_pos, start_pos),
                ]
            ):
                return True

        # If no specific positions are provided, check for general repetition
        elif len(relevant_moves) >= 4:
            last_move = relevant_moves[-1]
            if (
                relevant_moves[-4] == relevant_moves[-2] == last_move and
                relevant_moves[-3] == relevant_moves[-1]
            ):
                return True

        return False

    ## Function to place a goat on the board
    def place_goat(self, position):
        """
        Place a goat on the board at the specified position.

        :param position: Tuple (row, col) where the goat will be placed.
        :return: Boolean indicating success of placement.
        """
        row, col = position

        # Check if the position is valid for placement
        if 0 <= row < 5 and 0 <= col < 5 and self.board[row][col] == 0:
            self.board[row][col] = 2  # Place a goat
            self.goats_placed += 1
            return True
        else:
            print("Invalid placement. Try a different position.")
            return False
        
    def is_goat_placement_phase(self):
        """
        Check if the game is still in the goat placement phase.

        :return: Boolean indicating if goats are still being placed.
        """
        return self.goats_placed < 20
    
    ## Check valid moves for a piece
    def get_valid_moves(self, piece_type):
        """
        Get all valid moves for a given piece type (tiger or goat).
        """
        valid_moves = []

        if self.is_goat_placement_phase() and piece_type == 2:  # Goat placement phase
            # Goats can be placed on any empty cell
            for row in range(5):
                for col in range(5):
                    if self.board[row][col] == 0:
                        valid_moves.append(((row, col),))  # Placement only needs the target cell
        else:
            # Standard movement phase logic
            for row in range(5):
                for col in range(5):
                    if self.board[row][col] == piece_type:
                        # Get all valid neighbors for this piece
                        for neighbor in self.adjacency_map.get((row, col), []):
                            n_row, n_col = neighbor
                            # Check if the neighboring cell is empty
                            if self.board[n_row][n_col] == 0:
                                valid_moves.append(((row, col), (n_row, n_col)))

                            # Check for tiger captures
                            if piece_type == 1:  # Tiger
                                jump_row = n_row + (n_row - row)
                                jump_col = n_col + (n_col - col)

                                # Ensure the jump target is within bounds and valid
                                if 0 <= jump_row < 5 and 0 <= jump_col < 5:
                                    if self.board[n_row][n_col] == 2 and self.board[jump_row][jump_col] == 0:
                                        valid_moves.append(((row, col), (jump_row, jump_col)))
        return valid_moves

    ## Function to make a move
    def make_move(self, start_pos, end_pos):
        """
        Apply a move to the board and update the state, with repetitive move prevention.
        """
        piece_type = "tiger" if self.board[start_pos[0]][start_pos[1]] == 1 else "goat"

        # Check for repetitive move pattern
        if self.is_repetitive_move(piece_type, start_pos, end_pos):
            print("Invalid move: Repetitive moves are not allowed. Choose a different move.")
            return False

        # Move the piece
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = 0

        # If the move is valid, add it to the move history
        self.move_history.append((piece_type, start_pos, end_pos))

        # Check if a goat is captured (tiger jumps)
        if abs(end_row - start_row) == 2 or abs(end_col - start_col) == 2:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            self.board[mid_row][mid_col] = 0
            self.captured_goats += 1

        return True

    ## Function to check victory conditions
    ## Check if the game has ended and determine the winner.
    def check_victory_conditions(self):
        """
        Check if the game has ended and determine the winner.

        :return: String indicating the game status: 'ongoing', 'tiger_win', or 'goat_win'.
        """
        # Check for tiger victory (5 goats captured)
        if self.captured_goats >= 5:
            self.status = "tiger_win"
            return self.status
        
        # Check if all tigers are blocked (no valid moves for tigers, goat wins)
        if len(self.get_valid_moves(1)) == 0:
            self.status = "goat_win"
            return self.status
        
        # Check for repetitive state (draw)
        if self.is_repetitive_state():
            self.status = "draw"
            return self.status

        # Game is still ongoing
        self.status = "ongoing"
        return self.status
    
    ## Function to display the board
    def display_board(self):
        """
        Display the current state of the Bagh Chal board.
        """
        for row in self.board:
            print(" ".join(["T" if cell == 1 else "G" if cell == 2 else "." for cell in row]))
        print()
    
    ## Function to evaluate the state of the game
    def evaluate_state(self):
        """
        Evaluate the current game state and return a score.
        Positive score favors tigers, negative score favors goats.
        :return: A numerical score representing the game state.
        """
        # Determine the current phase
        in_placement_phase = self.is_goat_placement_phase()

        # Initialize scores
        mobility_score = 0
        capture_potential = 0
        blocking_score = 0
        captured_score = 0
        goat_survival_score = 0
        repetition_penalty = 0

        # Factor 1: Mobility
        tiger_moves = len(self.get_valid_moves(piece_type=1))
        goat_moves = len(self.get_valid_moves(piece_type=2)) if not in_placement_phase else 0
        mobility_score = (tiger_moves * 4) - (goat_moves * 3)  # Adjusted weights

        # Factor 2: Capture Potential (only in movement phase)
        if not in_placement_phase:
            for tiger_move in self.get_valid_moves(piece_type=1):
                start, end = tiger_move
                mid_row = (start[0] + end[0]) // 2
                mid_col = (start[1] + end[1]) // 2
                if self.board[mid_row][mid_col] == 2:  # Goat in jump position
                    capture_potential += 12  # Fine-tuned reward for potential captures

        # Factor 3: Blocking Effectiveness
        blocked_tigers = 0
        for tiger_pos in [(r, c) for r in range(5) for c in range(5) if self.board[r][c] == 1]:
            if all(self.board[n_row][n_col] != 0 for n_row, n_col in self.adjacency_map[tiger_pos]):
                blocked_tigers += 1
        blocking_score = -25 * blocked_tigers  # Slightly increased penalty for blocked tigers

        # Factor 4: Captured Goats
        captured_score = self.captured_goats * 35  # Slightly increased reward for captured goats

        # Factor 5: Remaining Goats
        remaining_goats = 20 - self.captured_goats
        goat_survival_score = remaining_goats * -1.5  # Reduced penalty for goat survival

        # Factor 6: Repetition Penalty
        if self.is_repetitive_move("tiger", None, None):  # Check for tiger's repetitive moves
            repetition_penalty -= 40  # Apply penalty for tiger

        if self.is_repetitive_move("goat", None, None):  # Check for goat's repetitive moves
            repetition_penalty -= 20  # Apply penalty for goat

        # Dynamic Scaling: Adjust weights near game-end
        if self.captured_goats >= 4:  # Tigers close to winning
            capture_potential *= 2
            captured_score *= 1.5
        if blocked_tigers >= 3:  # Goats close to winning
            blocking_score *= 2

        # Check game termination conditions
        if self.status == "tiger_win":
            return 1000  # Large positive score for tiger win
        elif self.status == "goat_win":
            return -1000  # Large negative score for goat win

        # Combine factors
        score = (
            mobility_score +
            capture_potential +
            blocking_score +
            captured_score +
            goat_survival_score +
            repetition_penalty
        )
        return score

    def serialize_state_binary(self):
        """
        Serialize the current game state into a compact binary representation.
        :return: A tuple containing the binary state (board as integer, goats_placed, captured_goats, status).
        Improved state representation with game phase and player turn information.
        """
        # Encode the board as a single integer
        # Basic board state
        board_state = np.zeros((5, 5, 5), dtype=np.uint8)  # 5x5x5 tensor
        for row in range(5):
            for col in range(5):
                cell = self.board[row][col]
                if cell == 1:
                    board_state[row,col,0] = 1  # Tiger layer
                elif cell == 2:
                    board_state[row,col,1] = 1  # Goat layer
                else:
                    board_state[row,col,2] = 1  # Empty layer

        # Map status to an integer for compact storage
        # Layer 3: Current player (1 for tiger, 0 for goat)
        board_state[:, :,3] = 1 if self.current_player() == "tiger" else 0  # Player turn
        board_state[:, :,4] = (20 - self.goats_placed) / 20  # Goats remaining
        
        return board_state
    
    def deserialize_state_binary(self, serialized_state):
        """
        Deserialize a binary state representation and update the game state.
        :param serialized_state: A 5x5x5 numpy array containing the binary representation.
        """
        # Decode the board from the 5x5x5 tensor
        self.board = [[0] * 5 for _ in range(5)]
        for row in range(5):
            for col in range(5):
                if serialized_state[row, col, 0] == 1:
                    self.board[row][col] = 1
                elif serialized_state[row, col, 1] == 1:
                    self.board[row][col] = 2

        # Reconstruct game state from the tensor
        self.goats_placed = 20 - int(serialized_state[0, 0, 4] * 20)  # Goats placed
        self.status = "ongoing"  # Reset status to ongoing

    ## Function to simulate random action during process
    def simulate_action(self, state, action):
        """Simulate action without modifying actual game state"""
        # Create temporary game copy
        temp_game = copy.deepcopy(self)
        temp_game.deserialize_state_binary(state)
        
        try:
            if len(action) == 1:  # Goat placement
                if not temp_game.place_goat(action[0]):
                    raise ValueError(f"Invalid goat placement at {action[0]}")
            else:  # Movement
                if not temp_game.make_move(action[0], action[1]):
                    raise ValueError(f"Invalid move from {action[0]} to {action[1]}")
        except Exception as e:
            print(f"Simulation failed: {str(e)}")
            return state  # Return original state on failure
        
        return temp_game.serialize_state_binary()
    
    # ADDED HELPER METHOD
    def get_current_state(self):
        """Get complete game state as a dictionary (for debugging)"""
        return {
            "board": [row.copy() for row in self.board],
            "goats_placed": self.goats_placed,
            "captured_goats": self.captured_goats,
            "status": self.status,
            "player": self.current_player()
        }
        
    def current_player(self):
        """
        Determine the current player based on the turn or phase.
        :return: 'goat' or 'tiger' based on the current phase.
        """
        if self.is_goat_placement_phase():
            return "goat"   # Goat placement phase
        else:
            # Determine the current player based on the number of moves made
            moves_made = len(self.move_history)
            current = "tiger" if moves_made % 2 == 0 else "goat"
            return current
        
    def reset(self):
        """
        Reset the game to its initial state.
        """
        self.board = [
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
        ]
        self.goats_placed = 0
        self.captured_goats = 0
        self.status = "ongoing"
        self.move_history = []
        self.state_history.clear()

# Example usage
if __name__ == "__main__":
    game = BaghChalGame()

    # Serialize initial state
    serialized_state = game.serialize_state_binary()
    print("Serialized State (Initial):", serialized_state)

    # Deserialize and validate
    game.deserialize_state_binary(serialized_state)
    print("Deserialized Board:")
    game.display_board()

