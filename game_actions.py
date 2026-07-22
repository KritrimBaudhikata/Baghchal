"""
Canonical action encoding for Baghchal.
The same action catalog is shared by rules, MCTS, self-play, and neural-network heads.
"""

from baghchal import BaghChalGame


def _build_action_catalog():
    game = BaghChalGame()
    actions = []

    for row in range(5):
        for col in range(5):
            actions.append(((row, col),))

    for row in range(5):
        for col in range(5):
            start = (row, col)
            for neighbor in game.adjacency_map.get(start, []):
                actions.append((start, neighbor))
                jump_row = neighbor[0] + (neighbor[0] - row)
                jump_col = neighbor[1] + (neighbor[1] - col)
                jump_target = (jump_row, jump_col)
                if 0 <= jump_row < 5 and 0 <= jump_col < 5:
                    if jump_target in game.adjacency_map.get(neighbor, []):
                        actions.append((start, jump_target))

    unique = []
    seen = set()
    for action in actions:
        if action in seen:
            continue
        seen.add(action)
        unique.append(action)
    return unique


ACTION_CATALOG = _build_action_catalog()
ACTION_SPACE_SIZE = len(ACTION_CATALOG)
_ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTION_CATALOG)}


def action_to_index(action):
    """
    Map an action tuple to a stable index.
    """
    return _ACTION_TO_INDEX[action]


def index_to_action(index):
    """
    Map an action index back to its tuple representation.
    """
    return ACTION_CATALOG[index]
