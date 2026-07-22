# Baghchal – Updates and Next Steps

This document summarizes the updates made (from the playable-game and hosting plan) and recommended next steps. It also answers whether you can delete the **old code** and **notebooks** folders.

---

## Updates Made

### Phase A – Correctness

| Change | Location | Description |
|--------|----------|-------------|
| Shared action encoding | **`game_actions.py`** (new) | `action_to_index(action)` moved here so MCTS and self-play use the same 0–64 mapping. Prevents mask/prior index drift. |
| MCTS action mask and prior | **`training/mcts.py`** | Mask and child prior now use `action_to_index(action)` instead of `enumerate(valid_actions)`. Added `_select()` wrapper for selection. |
| State deserialization | **`baghchal.py`** | `deserialize_state_binary` now sets `captured_goats` (from goats_placed − goats_on_board), and clears `move_history` / `state_history`. |
| Serialization fix | **`baghchal.py`** | Layer 4 no longer stores a fraction in uint8 (which truncated to 0). It now stores `goats_placed` (0–20) in `[0,0,4]` so state round-trips correctly and avoids false "tiger_win". |
| Copy import | **`models/utils.py`** | `import copy` moved to the top (was at bottom). |

---

### Phase B – Environment and Model I/O

| Change | Location | Description |
|--------|----------|-------------|
| Version pins | **`requirements.txt`** | Major version ranges (e.g. `tensorflow>=2.10,<2.20`, `numpy>=1.24,<2`) and PyYAML. |
| API deps | **`requirements.txt`** | Added `fastapi`, `uvicorn[standard]`. |
| Env ignore | **`.gitignore`** | `.venv/`, `venv/`, `env/`, `.conda/`, `*.h5`. |
| Conda env | **`environment.yml`** (new) | Optional conda env with Python 3.11 and pip install from requirements. |
| Config | **`config.yaml`** (new) | Training (iterations, batch size, sims, save_dir, mcts_batch_size, parallel_workers), evaluation, paths, API port. |
| Config loader | **`config_loader.py`** (new) | Loads `config.yaml` and applies env overrides (`BAGHCHAL_MODEL_PATH`, `BAGHCHAL_SAVE_DIR`). |
| Model format | **`training/training_loop.py`** | Saves `.keras` (not `.h5`) and writes **`checkpoint_manifest.json`** (iteration, learning_rate, model_path). |
| Load helper | **`models/neural_network.py`** | `load_bagh_chal_model(path)` for `.keras` and legacy `.h5`. |
| Resume training | **`main.py`** | `--resume path/to/model.keras` loads checkpoint and continues from `checkpoint_manifest.json` iteration. Uses config for all training/eval settings. |
| Inference script | **`run_inference.py`** (new) | Loads a saved model and runs N games (e.g. AI vs random). Example: `python run_inference.py models/final_bagh_chal_model.keras --games 3`. |
| README | **`README.md`** | Venv/conda, GPU check, resume, inference, **Run API locally**, **Deploy to cloud**, and updated project structure. |

---

### Phase C – Efficiency

| Change | Location | Description |
|--------|----------|-------------|
| Batched MCTS | **`training/mcts.py`** | `batch_size` (default 32): collects leaf states, runs one `model.predict` per batch, expands each unique node once, then backpropagates all. Sequential path kept when `batch_size <= 1`. |
| Parallel self-play | **`training/self_play.py`** | `generate_self_play_data_parallel(game, model_path, num_workers=...)` runs games in subprocesses; each worker loads the model from path. Training loop uses it when `config.yaml` has `parallel_workers > 0` (saves current model as `_current_iter.keras` each iteration). |
| Config options | **`config.yaml`** | `mcts_batch_size`, `parallel_workers`. **`main.py`** and **`training/training_loop.py`** read and pass these through. |

---

### Phase D – Web and Hosting

| Change | Location | Description |
|--------|----------|-------------|
| FastAPI backend | **`api/app.py`** (new) | Loads model at startup from `BAGHCHAL_MODEL_PATH` or `models/final_bagh_chal_model.keras`. Endpoints: `POST /api/game`, `GET /api/game/{id}`, `POST /api/game/{id}/move`. In-memory games; after each human (Goat) move, AI (Tiger) move is computed with MCTS and applied. |
| Web frontend | **`api/static/index.html`** (new) | Single-page 5×5 board: New game, place goat, select goat and click destination to move. Shows “Thinking…” while API runs MCTS. |
| Package | **`api/__init__.py`** (new) | Makes `api` a package. |
| Docker | **`Dockerfile`** (new) | Python 3.11-slim, install deps, copy app and model dir; serve on 8080. |
| Docker ignore | **`.dockerignore`** (new) | Excludes .git, venv, __pycache__, notebooks, etc. |
| README | **`README.md`** | “Run API locally” and “Deploy to cloud (Railway, Render, Fly.io)” with commands and env vars. |

---

## Next Steps

1. **Install dependencies**  
   `pip install -r requirements.txt` (includes FastAPI/uvicorn for the API).

2. **Confirm GPU**  
   `python test_gpu.py` to ensure TensorFlow sees your laptop GPU for training.

3. **Train (or resume)**  
   - New run: `python main.py`  
   - Resume: `python main.py --resume models/bagh_chal_model_iter_5.keras`  
   Tune `config.yaml` (e.g. `num_iterations`, `games_per_iteration`, `parallel_workers`, `mcts_batch_size`).

4. **Run inference**  
   `python run_inference.py models/final_bagh_chal_model.keras --games 5` to validate a checkpoint.

5. **Run API locally**  
   Set `BAGHCHAL_MODEL_PATH` to your `.keras` model, then:  
   `uvicorn api.app:app --reload --port 8080`  
   Open http://localhost:8080 and play as Goats vs AI (Tigers).

6. **Deploy to cloud**  
   Build the Docker image, ensure the trained model is in `models/` (or mounted), and deploy to Railway, Render, or Fly.io with port 8080 and the default uvicorn command (see README).

7. **Optional improvements**  
   - Reduce `copy.deepcopy` in MCTS (e.g. state stack / undo in the game).  
   - Use `tf.data.Dataset` in the training loop for prefetch.  
   - Add a smaller “fast” model for the web app if the full model is too heavy on the chosen cloud instance.

---

## Remaining from the plan (optional or deferred)

Required plan items for playable game + hosting are done. Status of follow-ups:

| Item | Plan ref | Status | Notes |
|------|----------|--------|--------|
| Fewer deepcopies in MCTS | Sec 4 | **Done** | clone/snapshot/restore; MCTS stores full snapshots. |
| tf.data.Dataset for training | Sec 4 optional | **Done** | build_tf_dataset with prefetch; Sequence fallback. |
| CI (GitHub Actions) | Sec 6 optional | **Done** | .github/workflows/ci.yml |
| Exact version lock | Sec 2 | **Done** | requirements.txt pins + requirements-lock.txt |
| Replay buffer resume | Sec 3 | Skipped | v1 empty buffer on resume |
| move_history/state_history | Sec 1.2, 10 | **Done** | Via snapshot API used by MCTS |
| index_to_action | Sec 10 optional | **Done** | In game_actions.py |
| Docker non-root user | Sec 8 | **Done** | User baghchal in Dockerfile |
| Smaller fast model for web | Sec 10 | Not built | Optional if cloud RAM is tight |
| API AI failure recovery | status follow-up | **Done** | Fallback + /ai-move + ai_pending |

No required plan items are missing. Optional next: train a fresh model (ACTION_SPACE_SIZE 215), then deploy.

---

## Can You Delete “old code” and “notebooks”?

### “old code” folder

- **Referenced in `.gitignore`** as `/old code` and `/old code.rar` (so it’s treated as local/backup).
- **No Python code imports** from `old code` anywhere in the project.
- **Verdict:** **Yes, you can delete the contents (or the whole folder)** if you no longer need that backup. It’s legacy; the current app and training pipeline do not depend on it.

### “notebooks” folder

- Contains **`exploration.ipynb`** and **`test.ipynb`**.
- **No Python code imports** these notebooks or any module inside `notebooks/`.
- They are for **ad-hoc exploration and experiments**, not for running the main app, training, or API.
- **Verdict:** **Yes, you can delete the contents (or the whole folder)** if you don’t need those experiments. Optional to keep for reference; deleting them does not affect training, inference, or the web game.

**Summary:** Both **old code** and **notebooks** are safe to delete from a “does the project still run?” perspective. Delete them if you want to free space and don’t need the backup or notebook experiments.
