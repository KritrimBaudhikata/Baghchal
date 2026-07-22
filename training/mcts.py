import math
import numpy as np

from baghchal import BaghChalGame
from game_actions import ACTION_SPACE_SIZE, action_to_index


class MCTSNode:
    def __init__(self, state, valid_actions, parent=None, action=None):
        self.state = state  # full snapshot dict (includes state_history)
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0
        self.valid_actions = valid_actions

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_fully_expanded(self):
        return len(self.children) == len(self.valid_actions)


class MCTS:
    def __init__(self, game, model, c_puct=1.25, num_simulations=200, batch_size=32):
        self.base_game = game
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.batch_size = batch_size

    def _empty_mask(self):
        return np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

    def _mask_for_actions(self, actions):
        mask = self._empty_mask()
        for action in actions:
            idx = action_to_index(action)
            mask[idx] = 1.0
        return mask

    def _terminal_value(self, game):
        if game.status == "tiger_win":
            winner = "tiger"
        elif game.status == "goat_win":
            winner = "goat"
        else:
            return 0.0
        return 1.0 if game.current_player() == winner else -1.0

    def _normalize_root(self, root_state):
        """Accept a live game, snapshot dict, or legacy NN tensor."""
        if isinstance(root_state, BaghChalGame):
            return root_state.snapshot()
        if isinstance(root_state, dict) and "board" in root_state:
            return root_state
        game = self.base_game.clone()
        game.deserialize_state_binary(root_state)
        return game.snapshot()

    def _game_from_snapshot(self, snapshot):
        return BaghChalGame.from_snapshot(snapshot, adjacency_map=self.base_game.adjacency_map)

    def search(self, root_state):
        root_snapshot = self._normalize_root(root_state)
        root_game = self._game_from_snapshot(root_snapshot)
        root = MCTSNode(root_snapshot, root_game.legal_actions())

        if self.batch_size <= 1:
            return self._search_sequential(root_snapshot, root)

        pending = []
        for _ in range(self.num_simulations):
            node, leaf_game = self._select_leaf(root, root_snapshot)

            if leaf_game.check_victory_conditions() != "ongoing":
                self._backpropagate(node, self._terminal_value(leaf_game))
                continue

            if node.is_fully_expanded():
                self._backpropagate(node, 0.0)
                continue

            state_np = leaf_game.serialize_state_binary()
            valid_actions = leaf_game.legal_actions()
            mask_np = self._mask_for_actions(valid_actions)
            pending.append((node, state_np, mask_np))

            if len(pending) >= self.batch_size:
                self._process_batch(pending)
                pending = []

        if pending:
            self._process_batch(pending)
        return root

    def _select_leaf(self, root, root_snapshot):
        node = root
        temp_game = self._game_from_snapshot(root_snapshot)

        while node.is_fully_expanded() and node.children:
            _, node = self._select(node)
            temp_game.restore(node.state)
        return node, temp_game

    def _process_batch(self, pending):
        states = np.array([entry[1] for entry in pending])
        masks = np.array([entry[2] for entry in pending])
        policy_batch, value_batch = self.model.predict([states, masks], verbose=0)

        expanded_nodes = set()
        for index, (node, _state_np, _mask_np) in enumerate(pending):
            if id(node) in expanded_nodes:
                continue
            expanded_nodes.add(id(node))
            self._expand_with_policy(node, policy_batch[index])

        for index, (node, _state_np, _mask_np) in enumerate(pending):
            value = float(value_batch[index][0])
            self._backpropagate(node, value)

    def _expand_with_policy(self, node, policy):
        game = self._game_from_snapshot(node.state)
        if game.check_victory_conditions() != "ongoing":
            return

        for action in game.legal_actions():
            child_state, child_game = game.next_state_from_action(node.state, action)
            if child_state is None or child_game is None:
                continue
            child = MCTSNode(
                state=child_state,
                valid_actions=child_game.legal_actions(),
                parent=node,
                action=action,
            )
            child.prior = float(policy[action_to_index(action)])
            node.children[action] = child

    def _search_sequential(self, root_snapshot, root):
        for _ in range(self.num_simulations):
            node, temp_game = self._select_leaf(root, root_snapshot)

            if temp_game.check_victory_conditions() != "ongoing":
                self._backpropagate(node, self._terminal_value(temp_game))
                continue

            if not node.is_fully_expanded():
                leaf_value = self._expand(node)
                self._backpropagate(node, leaf_value)
                continue

            leaf_value = self._simulate(temp_game)
            self._backpropagate(node, leaf_value)
        return root

    def _select_child(self, node):
        total_n = math.sqrt(max(1, sum(child.visit_count for child in node.children.values())))
        best_score = -np.inf
        best_action = None
        best_child = None

        for action, child in node.children.items():
            exploration = self.c_puct * child.prior * total_n / (1 + child.visit_count)
            score = child.q_value + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _select(self, node):
        return self._select_child(node)

    def _expand(self, node):
        game = self._game_from_snapshot(node.state)
        if game.check_victory_conditions() != "ongoing":
            return self._terminal_value(game)

        valid_actions = game.legal_actions()
        mask = self._mask_for_actions(valid_actions)
        state_input = game.serialize_state_binary()[np.newaxis, ...]
        policy, value = self.model.predict([state_input, mask[np.newaxis, ...]], verbose=0)

        for action in valid_actions:
            child_state, child_game = game.next_state_from_action(node.state, action)
            if child_state is None or child_game is None:
                continue
            child = MCTSNode(
                state=child_state,
                valid_actions=child_game.legal_actions(),
                parent=node,
                action=action,
            )
            child.prior = float(policy[0][action_to_index(action)])
            node.children[action] = child
        return float(value[0][0])

    def _simulate(self, temp_game):
        if temp_game.check_victory_conditions() != "ongoing":
            return self._terminal_value(temp_game)
        state_input = temp_game.serialize_state_binary()[np.newaxis, ...]
        action_mask = np.ones(ACTION_SPACE_SIZE, dtype=np.float32)
        _, value = self.model.predict([state_input, action_mask[np.newaxis, ...]], verbose=0)
        return float(value[0][0])

    def _backpropagate(self, node, value):
        while node:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent
