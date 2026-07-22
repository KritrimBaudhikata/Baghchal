import hashlib
import numpy as np


class BaghChalGame:
    STATUS_CODES = {"ongoing": 0, "tiger_win": 1, "goat_win": 2, "draw": 3}
    STATUS_BY_CODE = {value: key for key, value in STATUS_CODES.items()}

    def __init__(self):
        self.board = [
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
        ]
        self.goats_placed = 0
        self.captured_goats = 0
        self.adjacency_map = self._generate_bagh_chal_adjacency_map()
        self.status = "ongoing"
        self.turn_player = "goat"
        self.phase = "placement"
        self.ply_count = 0
        self.move_history = []
        self.state_history = {}
        self._record_state_occurrence()

    def _state_signature(self, state: np.ndarray) -> bytes:
        state_view = state.astype(np.uint8, copy=False)
        h = hashlib.blake2b(digest_size=16)
        h.update(state_view.tobytes())
        return h.digest()

    def _record_state_occurrence(self):
        signature = self._state_signature(self.serialize_state_binary())
        self.state_history[signature] = self.state_history.get(signature, 0) + 1

    def is_repetitive_state(self):
        if self.goats_placed < 20:
            return False
        signature = self._state_signature(self.serialize_state_binary())
        return self.state_history.get(signature, 0) >= 3

    def _generate_bagh_chal_adjacency_map(self):
        """5x5 Baghchal graph: all orthogonal links plus X-diagonals on even (r+c) 2x2 cells."""
        adjacency = {(r, c): [] for r in range(5) for c in range(5)}
        for row in range(5):
            for col in range(5):
                for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    n_row, n_col = row + d_row, col + d_col
                    if 0 <= n_row < 5 and 0 <= n_col < 5:
                        adjacency[(row, col)].append((n_row, n_col))
        for row in range(4):
            for col in range(4):
                if (row + col) % 2 != 0:
                    continue
                tl, br = (row, col), (row + 1, col + 1)
                tr, bl = (row, col + 1), (row + 1, col)
                adjacency[tl].append(br)
                adjacency[br].append(tl)
                adjacency[tr].append(bl)
                adjacency[bl].append(tr)
        return {pos: sorted(set(neighbors)) for pos, neighbors in adjacency.items()}

    def _in_bounds(self, row, col):
        return 0 <= row < 5 and 0 <= col < 5

    def _update_phase(self):
        self.phase = "placement" if self.goats_placed < 20 else "movement"

    def _piece_for_player(self, player):
        return 1 if player == "tiger" else 2

    def _is_tiger_capture(self, start_pos, end_pos):
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        delta_row = end_row - start_row
        delta_col = end_col - start_col
        if delta_row == 0 and delta_col == 0:
            return False
        if delta_row % 2 != 0 or delta_col % 2 != 0:
            return False

        mid_row = start_row + delta_row // 2
        mid_col = start_col + delta_col // 2
        if not self._in_bounds(mid_row, mid_col):
            return False
        if not self._in_bounds(end_row, end_col):
            return False

        mid_pos = (mid_row, mid_col)
        if mid_pos not in self.adjacency_map.get(start_pos, []):
            return False
        if end_pos not in self.adjacency_map.get(mid_pos, []):
            return False
        return self.board[mid_row][mid_col] == 2 and self.board[end_row][end_col] == 0

    def _movement_actions_for_piece(self, piece_type):
        actions = []
        for row in range(5):
            for col in range(5):
                if self.board[row][col] != piece_type:
                    continue
                start = (row, col)
                for neighbor in self.adjacency_map.get(start, []):
                    n_row, n_col = neighbor
                    if self.board[n_row][n_col] == 0:
                        actions.append((start, neighbor))
                    elif piece_type == 1 and self.board[n_row][n_col] == 2:
                        jump_row = n_row + (n_row - row)
                        jump_col = n_col + (n_col - col)
                        jump = (jump_row, jump_col)
                        if self._in_bounds(jump_row, jump_col) and self._is_tiger_capture(start, jump):
                            actions.append((start, jump))
        return actions

    def legal_actions(self):
        current_piece = self._piece_for_player(self.turn_player)
        return self.get_valid_moves(current_piece, respect_turn=True)

    def get_valid_moves(self, piece_type, respect_turn=True):
        if self.status != "ongoing":
            return []
        if respect_turn and piece_type != self._piece_for_player(self.turn_player):
            return []

        if piece_type == 2 and self.goats_placed < 20:
            placements = []
            for row in range(5):
                for col in range(5):
                    if self.board[row][col] == 0:
                        placements.append(((row, col),))
            return placements

        if piece_type == 2 and self.goats_placed < 20:
            return []
        return self._movement_actions_for_piece(piece_type)

    def _apply_placement(self, position):
        row, col = position
        if not self._in_bounds(row, col):
            return False
        if self.board[row][col] != 0:
            return False
        self.board[row][col] = 2
        self.goats_placed += 1
        self.move_history.append(("goat", None, position))
        return True

    def _apply_movement(self, start_pos, end_pos):
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        if not (self._in_bounds(start_row, start_col) and self._in_bounds(end_row, end_col)):
            return False
        piece = self.board[start_row][start_col]
        if piece == 0:
            return False
        if self.board[end_row][end_col] != 0:
            return False
        is_capture = piece == 1 and self._is_tiger_capture(start_pos, end_pos)
        if end_pos not in self.adjacency_map.get(start_pos, []) and not is_capture:
            return False

        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = 0

        if is_capture:
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            self.board[mid_row][mid_col] = 0
            self.captured_goats += 1

        moving_player = "tiger" if piece == 1 else "goat"
        self.move_history.append((moving_player, start_pos, end_pos))
        return True

    def apply_action(self, action):
        if self.status != "ongoing":
            return False
        legal = self.legal_actions()
        if action not in legal:
            return False

        if len(action) == 1:
            if self.turn_player != "goat":
                return False
            ok = self._apply_placement(action[0])
        else:
            start_pos, end_pos = action
            start_piece = self.board[start_pos[0]][start_pos[1]]
            if start_piece != self._piece_for_player(self.turn_player):
                return False
            ok = self._apply_movement(start_pos, end_pos)

        if not ok:
            return False

        self.ply_count += 1
        self.turn_player = "goat" if self.turn_player == "tiger" else "tiger"
        self._update_phase()
        self._record_state_occurrence()
        self.check_victory_conditions()
        return True

    def is_goat_placement_phase(self):
        return self.goats_placed < 20

    def place_goat(self, position):
        return self.apply_action((position,))

    def make_move(self, start_pos, end_pos):
        return self.apply_action((start_pos, end_pos))

    def is_repetitive_move(self, piece_type, start_pos=None, end_pos=None):
        if len(self.move_history) < 4:
            return False
        relevant_moves = [move for move in self.move_history if move[0] == piece_type]
        if len(relevant_moves) < 4:
            return False
        if start_pos is not None and end_pos is not None:
            return relevant_moves[-4:] == [
                (piece_type, start_pos, end_pos),
                (piece_type, end_pos, start_pos),
                (piece_type, start_pos, end_pos),
                (piece_type, end_pos, start_pos),
            ]
        return False

    def check_victory_conditions(self):
        if self.captured_goats >= 5:
            self.status = "tiger_win"
            return self.status

        tiger_actions = self.get_valid_moves(1, respect_turn=False)
        if not tiger_actions:
            self.status = "goat_win"
            return self.status

        # ponytail: goat immobilized ≠ tiger win in classical rules; draw unblocks self-play/API
        if self.turn_player == "goat" and self.goats_placed >= 20 and not self.get_valid_moves(2, respect_turn=False):
            self.status = "draw"
            return self.status

        if self.is_repetitive_state():
            self.status = "draw"
            return self.status

        self.status = "ongoing"
        return self.status

    def display_board(self):
        for row in self.board:
            print(" ".join(["T" if cell == 1 else "G" if cell == 2 else "." for cell in row]))
        print()

    def evaluate_state(self):
        in_placement_phase = self.is_goat_placement_phase()
        mobility_score = 0
        capture_potential = 0
        blocking_score = 0
        captured_score = 0
        goat_survival_score = 0
        repetition_penalty = 0

        tiger_moves = len(self.get_valid_moves(piece_type=1, respect_turn=False))
        goat_moves = len(self.get_valid_moves(piece_type=2, respect_turn=False)) if not in_placement_phase else 0
        mobility_score = (tiger_moves * 4) - (goat_moves * 3)

        if not in_placement_phase:
            for tiger_move in self.get_valid_moves(piece_type=1, respect_turn=False):
                if len(tiger_move) != 2:
                    continue
                start, end = tiger_move
                if self._is_tiger_capture(start, end):
                    capture_potential += 12

        blocked_tigers = 0
        for tiger_pos in [(r, c) for r in range(5) for c in range(5) if self.board[r][c] == 1]:
            if all(self.board[n_row][n_col] != 0 for n_row, n_col in self.adjacency_map[tiger_pos]):
                blocked_tigers += 1
        blocking_score = -25 * blocked_tigers
        captured_score = self.captured_goats * 35

        remaining_goats = 20 - self.captured_goats
        goat_survival_score = remaining_goats * -1.5

        if self.is_repetitive_move("tiger"):
            repetition_penalty -= 40
        if self.is_repetitive_move("goat"):
            repetition_penalty -= 20

        if self.captured_goats >= 4:
            capture_potential *= 2
            captured_score *= 1.5
        if blocked_tigers >= 3:
            blocking_score *= 2

        if self.status == "tiger_win":
            return 1000
        if self.status == "goat_win":
            return -1000

        return (
            mobility_score
            + capture_potential
            + blocking_score
            + captured_score
            + goat_survival_score
            + repetition_penalty
        )

    def serialize_state_binary(self):
        board_state = np.zeros((5, 5, 5), dtype=np.uint8)
        for row in range(5):
            for col in range(5):
                cell = self.board[row][col]
                if cell == 1:
                    board_state[row, col, 0] = 1
                elif cell == 2:
                    board_state[row, col, 1] = 1
                else:
                    board_state[row, col, 2] = 1

        board_state[:, :, 3] = 1 if self.turn_player == "tiger" else 0
        board_state[0, 0, 4] = min(20, int(self.goats_placed))
        board_state[0, 1, 4] = min(20, int(self.captured_goats))
        board_state[0, 2, 4] = 0 if self.phase == "placement" else 1
        board_state[0, 3, 4] = self.STATUS_CODES.get(self.status, 0)
        board_state[0, 4, 4] = int(self.ply_count) % 255
        return board_state

    def deserialize_state_binary(self, serialized_state):
        """
        Restore board/meta from the NN tensor.
        move_history / state_history are not present in the tensor; use snapshot()/restore()
        when full repetition context is required (e.g. MCTS).
        """
        self.board = [[0] * 5 for _ in range(5)]
        for row in range(5):
            for col in range(5):
                if serialized_state[row, col, 0] == 1:
                    self.board[row][col] = 1
                elif serialized_state[row, col, 1] == 1:
                    self.board[row][col] = 2

        self.goats_placed = min(20, int(serialized_state[0, 0, 4]))
        self.captured_goats = min(20, int(serialized_state[0, 1, 4]))
        encoded_phase = int(serialized_state[0, 2, 4])
        self.phase = "movement" if encoded_phase == 1 else "placement"
        status_code = int(serialized_state[0, 3, 4])
        self.status = self.STATUS_BY_CODE.get(status_code, "ongoing")
        self.turn_player = "tiger" if int(serialized_state[0, 0, 3]) == 1 else "goat"
        self.ply_count = int(serialized_state[0, 4, 4])
        self.move_history = []
        self.state_history = {}
        self._record_state_occurrence()

    def snapshot(self):
        """Full mutable state including repetition history (for MCTS / cloning)."""
        return {
            "board": [row[:] for row in self.board],
            "goats_placed": self.goats_placed,
            "captured_goats": self.captured_goats,
            "status": self.status,
            "turn_player": self.turn_player,
            "phase": self.phase,
            "ply_count": self.ply_count,
            "move_history": list(self.move_history),
            "state_history": dict(self.state_history),
        }

    def restore(self, snapshot):
        """Restore a snapshot produced by snapshot()."""
        self.board = [row[:] for row in snapshot["board"]]
        self.goats_placed = snapshot["goats_placed"]
        self.captured_goats = snapshot["captured_goats"]
        self.status = snapshot["status"]
        self.turn_player = snapshot["turn_player"]
        self.phase = snapshot["phase"]
        self.ply_count = snapshot["ply_count"]
        self.move_history = list(snapshot.get("move_history", []))
        self.state_history = dict(snapshot.get("state_history", {}))

    def clone(self):
        """Efficient copy that shares the static adjacency map."""
        cloned = BaghChalGame.__new__(BaghChalGame)
        cloned.adjacency_map = self.adjacency_map
        cloned.restore(self.snapshot())
        return cloned

    @staticmethod
    def from_snapshot(snapshot, adjacency_map=None):
        game = BaghChalGame.__new__(BaghChalGame)
        if adjacency_map is None:
            # Build once via a throwaway instance only when no shared map is provided
            game.adjacency_map = BaghChalGame()._generate_bagh_chal_adjacency_map()
        else:
            game.adjacency_map = adjacency_map
        game.restore(snapshot)
        return game

    def simulate_action(self, state, action):
        if isinstance(state, dict):
            temp_game = BaghChalGame.from_snapshot(state, adjacency_map=self.adjacency_map)
        else:
            temp_game = self.clone()
            temp_game.deserialize_state_binary(state)
        if not temp_game.apply_action(action):
            return state
        if isinstance(state, dict):
            return temp_game.snapshot()
        return temp_game.serialize_state_binary()

    def next_state_from_action(self, state, action):
        if isinstance(state, dict):
            temp_game = BaghChalGame.from_snapshot(state, adjacency_map=self.adjacency_map)
        else:
            temp_game = self.clone()
            temp_game.deserialize_state_binary(state)
        if not temp_game.apply_action(action):
            return None, None
        if isinstance(state, dict):
            return temp_game.snapshot(), temp_game
        return temp_game.serialize_state_binary(), temp_game

    def get_current_state(self):
        return {
            "board": [row.copy() for row in self.board],
            "goats_placed": self.goats_placed,
            "captured_goats": self.captured_goats,
            "status": self.status,
            "player": self.current_player(),
            "phase": self.phase,
            "ply_count": self.ply_count,
        }

    def current_player(self):
        return self.turn_player

    def reset(self):
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
        self.turn_player = "goat"
        self.phase = "placement"
        self.ply_count = 0
        self.move_history = []
        self.state_history = {}
        self._record_state_occurrence()


if __name__ == "__main__":
    game = BaghChalGame()
    serialized_state = game.serialize_state_binary()
    print("Serialized State (Initial):", serialized_state)
    game.deserialize_state_binary(serialized_state)
    print("Deserialized Board:")
    game.display_board()
