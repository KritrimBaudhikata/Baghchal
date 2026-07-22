"""
Baghchal web API: play against the AI. Loads model at startup, keeps games in memory.
"""
import os
import sys
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baghchal import BaghChalGame
from training.mcts import MCTS
from training.self_play import select_action

app = FastAPI(title="Baghchal API")

games = {}
_model = None
_mcts = None


def get_model():
    global _model, _mcts
    if _model is None:
        path = os.environ.get("BAGHCHAL_MODEL_PATH") or os.environ.get("MODEL_PATH")
        if not path or not os.path.isfile(path):
            path = os.path.join(ROOT, "models", "final_bagh_chal_model.keras")
        if not os.path.isfile(path):
            raise RuntimeError(f"No model found at {path}. Train a model first.")
        from models.neural_network import load_bagh_chal_model

        _model = load_bagh_chal_model(path)
        base_game = BaghChalGame()
        _mcts = MCTS(base_game, _model, num_simulations=50, batch_size=16)
    return _model, _mcts


def _session_to_json(session):
    game = session["game"]
    return {
        "board": [list(row) for row in game.board],
        "goats_placed": game.goats_placed,
        "captured_goats": game.captured_goats,
        "status": game.status,
        "current_player": game.current_player(),
        "human_side": session["human_side"],
        "ai_side": session["ai_side"],
        "ai_pending": bool(session.get("ai_pending", False)),
        "ai_error": session.get("ai_error"),
    }


def _action_to_json(action):
    if len(action) == 1:
        to_row, to_col = action[0]
        return {"type": "place", "to": {"row": to_row, "col": to_col}}
    (from_row, from_col), (to_row, to_col) = action
    return {
        "type": "move",
        "from": {"row": from_row, "col": from_col},
        "to": {"row": to_row, "col": to_col},
    }


def _json_to_action(body):
    if body.type == "place":
        if body.row is None or body.col is None:
            raise HTTPException(status_code=400, detail="place requires row and col")
        return ((body.row, body.col),)

    if body.from_row is None or body.from_col is None or body.to_row is None or body.to_col is None:
        raise HTTPException(status_code=400, detail="move requires from_row, from_col, to_row, to_col")
    return ((body.from_row, body.from_col), (body.to_row, body.to_col))


def _select_ai_action(game):
    _model_ref, mcts = get_model()
    mcts.base_game = game
    root = mcts.search(game)
    return select_action(root, temperature=0.1)


def _fallback_ai_action(game):
    legal = game.legal_actions()
    if not legal:
        return None
    return legal[0]


def _run_ai_turn(session, max_turns=1):
    """
    Run AI turn(s). On MCTS failure, fall back to a legal action so the game
    never sticks on the AI side (avoids human 409). Sets session['ai_pending']
    only if no legal move could be applied.
    """
    game = session["game"]
    session["ai_pending"] = False
    session["ai_error"] = None
    turns = 0
    while game.status == "ongoing" and game.current_player() == session["ai_side"] and turns < max_turns:
        action = None
        try:
            action = _select_ai_action(game)
        except Exception as error:
            session["ai_error"] = str(error)
            print(f"AI MCTS failed, using fallback: {error}")
            action = _fallback_ai_action(game)

        if action is None:
            action = _fallback_ai_action(game)

        if action is None or not game.apply_action(action):
            game.check_victory_conditions()
            if game.status == "ongoing" and game.current_player() == session["ai_side"]:
                session["ai_pending"] = True
                if not session.get("ai_error"):
                    session["ai_error"] = "AI could not select a legal move"
            break
        game.check_victory_conditions()
        turns += 1
        session["ai_pending"] = False
        session["ai_error"] = None
    return not session.get("ai_pending", False)


class NewGameRequest(BaseModel):
    human_side: Optional[str] = "goat"  # goat | tiger


class NewGameResponse(BaseModel):
    game_id: str
    board: list
    goats_placed: int
    captured_goats: int
    status: str
    current_player: str
    human_side: str
    ai_side: str


class MoveRequest(BaseModel):
    type: str  # place | move
    row: Optional[int] = None
    col: Optional[int] = None
    from_row: Optional[int] = None
    from_col: Optional[int] = None
    to_row: Optional[int] = None
    to_col: Optional[int] = None


@app.on_event("startup")
def startup():
    try:
        get_model()
    except Exception as error:
        print(f"Warning: Model not loaded at startup: {error}. Will load on first request.")


@app.post("/api/game", response_model=NewGameResponse)
def new_game(body: Optional[NewGameRequest] = None):
    human_side = (body.human_side if body else "goat") or "goat"
    human_side = human_side.lower()
    if human_side not in {"goat", "tiger"}:
        raise HTTPException(status_code=400, detail="human_side must be 'goat' or 'tiger'")

    session = {
        "game": BaghChalGame(),
        "human_side": human_side,
        "ai_side": "tiger" if human_side == "goat" else "goat",
        "ai_pending": False,
        "ai_error": None,
    }
    game_id = str(uuid.uuid4())
    games[game_id] = session

    if session["game"].current_player() == session["ai_side"]:
        _run_ai_turn(session, max_turns=1)

    payload = _session_to_json(session)
    payload["game_id"] = game_id
    return payload


@app.get("/api/game/{game_id}")
def get_game(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    return _session_to_json(games[game_id])


@app.get("/api/game/{game_id}/legal-moves")
def get_legal_moves(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    session = games[game_id]
    game = session["game"]
    legal_actions = game.legal_actions() if game.status == "ongoing" else []

    return {
        "status": game.status,
        "current_player": game.current_player(),
        "human_side": session["human_side"],
        "ai_side": session["ai_side"],
        "legal_moves": [_action_to_json(action) for action in legal_actions],
    }


@app.post("/api/game/{game_id}/move")
def play_move(game_id: str, body: MoveRequest):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    session = games[game_id]
    game = session["game"]
    if game.status != "ongoing":
        return _session_to_json(session)

    if game.current_player() != session["human_side"]:
        raise HTTPException(status_code=409, detail="It is not the human player's turn")

    action = _json_to_action(body)
    if not game.apply_action(action):
        raise HTTPException(status_code=400, detail="Invalid move")
    game.check_victory_conditions()

    if game.status == "ongoing" and game.current_player() == session["ai_side"]:
        _run_ai_turn(session, max_turns=1)

    return _session_to_json(session)


@app.post("/api/game/{game_id}/ai-move")
def retry_ai_move(game_id: str):
    """Retry the AI turn when ai_pending is set after a previous failure."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    session = games[game_id]
    game = session["game"]
    if game.status != "ongoing":
        return _session_to_json(session)

    if game.current_player() != session["ai_side"] and not session.get("ai_pending"):
        raise HTTPException(status_code=409, detail="It is not the AI player's turn")

    _run_ai_turn(session, max_turns=1)
    return _session_to_json(session)


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
