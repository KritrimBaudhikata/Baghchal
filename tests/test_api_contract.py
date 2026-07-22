import os
import sys
import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api.app as app_module


def _stub_ai_turn(session, max_turns=1):
    game = session["game"]
    turns = 0
    while game.status == "ongoing" and game.current_player() == session["ai_side"] and turns < max_turns:
        legal = game.legal_actions()
        if not legal:
            game.status = "draw"
            break
        game.apply_action(legal[0])
        game.check_victory_conditions()
        turns += 1


def _client_with_stubbed_ai(monkeypatch):
    monkeypatch.setattr(app_module, "_run_ai_turn", _stub_ai_turn)
    app_module.games.clear()
    return TestClient(app_module.app)


def test_new_game_defaults_to_human_goat(monkeypatch):
    client = _client_with_stubbed_ai(monkeypatch)
    response = client.post("/api/game", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["human_side"] == "goat"
    assert payload["ai_side"] == "tiger"
    assert payload["current_player"] == "goat"


def test_new_game_with_human_tiger_runs_initial_ai_turn(monkeypatch):
    client = _client_with_stubbed_ai(monkeypatch)
    response = client.post("/api/game", json={"human_side": "tiger"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["human_side"] == "tiger"
    assert payload["ai_side"] == "goat"
    assert payload["goats_placed"] == 1
    assert payload["current_player"] == "tiger"


def test_legal_moves_endpoint_returns_structured_moves(monkeypatch):
    client = _client_with_stubbed_ai(monkeypatch)
    create = client.post("/api/game", json={}).json()
    game_id = create["game_id"]
    response = client.get(f"/api/game/{game_id}/legal-moves")
    assert response.status_code == 200
    payload = response.json()
    assert payload["current_player"] == "goat"
    assert payload["human_side"] == "goat"
    assert len(payload["legal_moves"]) > 0
    assert payload["legal_moves"][0]["type"] == "place"


def test_move_endpoint_enforces_turn_and_validity(monkeypatch):
    client = _client_with_stubbed_ai(monkeypatch)
    create = client.post("/api/game", json={}).json()
    game_id = create["game_id"]

    bad_move = client.post(
        f"/api/game/{game_id}/move",
        json={"type": "move", "from_row": 0, "from_col": 0, "to_row": 0, "to_col": 1},
    )
    assert bad_move.status_code == 400

    good_place = client.post(f"/api/game/{game_id}/move", json={"type": "place", "row": 2, "col": 2})
    assert good_place.status_code == 200
    payload = good_place.json()
    assert payload["goats_placed"] >= 1
    assert payload["current_player"] in {"goat", "tiger"}


def test_move_endpoint_rejects_when_not_human_turn(monkeypatch):
    client = _client_with_stubbed_ai(monkeypatch)
    create = client.post("/api/game", json={"human_side": "tiger"}).json()
    game_id = create["game_id"]

    # Stubbed AI already moved once, so this should be human turn (tiger) now.
    # Force a non-human turn by making a legal tiger move, then immediately try another human move.
    legal = client.get(f"/api/game/{game_id}/legal-moves").json()["legal_moves"]
    tiger_move = next(move for move in legal if move["type"] == "move")
    first = client.post(
        f"/api/game/{game_id}/move",
        json={
            "type": "move",
            "from_row": tiger_move["from"]["row"],
            "from_col": tiger_move["from"]["col"],
            "to_row": tiger_move["to"]["row"],
            "to_col": tiger_move["to"]["col"],
        },
    )
    assert first.status_code == 200

    second = client.post(
        f"/api/game/{game_id}/move",
        json={
            "type": "move",
            "from_row": tiger_move["from"]["row"],
            "from_col": tiger_move["from"]["col"],
            "to_row": tiger_move["to"]["row"],
            "to_col": tiger_move["to"]["col"],
        },
    )
    assert second.status_code == 409


def test_ai_move_retry_endpoint_clears_pending(monkeypatch):
    client = _client_with_stubbed_ai(monkeypatch)
    create = client.post("/api/game", json={}).json()
    game_id = create["game_id"]

    response = client.post(f"/api/game/{game_id}/move", json={"type": "place", "row": 2, "col": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("ai_pending") is False

    app_module.games[game_id]["ai_pending"] = True
    app_module.games[game_id]["game"].turn_player = app_module.games[game_id]["ai_side"]
    retry = client.post(f"/api/game/{game_id}/ai-move")
    assert retry.status_code == 200
    retry_payload = retry.json()
    assert "ai_pending" in retry_payload
    assert retry_payload["current_player"] == "goat" or retry_payload["status"] != "ongoing"


def test_mcts_failure_falls_back_so_human_is_not_stuck(monkeypatch):
    """If MCTS raises, AI still plays a legal move; next human move must not 409."""
    app_module.games.clear()

    def boom(_game):
        raise RuntimeError("mcts exploded")

    monkeypatch.setattr(app_module, "_select_ai_action", boom)
    client = TestClient(app_module.app)

    create = client.post("/api/game", json={}).json()
    game_id = create["game_id"]
    moved = client.post(f"/api/game/{game_id}/move", json={"type": "place", "row": 2, "col": 2})
    assert moved.status_code == 200
    payload = moved.json()
    assert payload["current_player"] == "goat"
    assert payload.get("ai_pending") is False

    again = client.post(f"/api/game/{game_id}/move", json={"type": "place", "row": 2, "col": 3})
    assert again.status_code == 200
    assert again.json()["goats_placed"] >= 2
