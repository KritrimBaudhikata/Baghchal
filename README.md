# Baghchal AlphaZero AI

A sophisticated implementation of the AlphaZero algorithm for the traditional Nepali board game **Baghchal** (Tiger and Goats), designed to achieve an ELO rating of 2000+.

## 🎯 Project Overview

Baghchal is a strategic board game where 4 tigers attempt to capture 20 goats, while goats try to block and trap the tigers. This project implements the AlphaZero algorithm - a combination of deep neural networks and Monte Carlo Tree Search (MCTS) - to create an AI that can play at expert level.

**Target Achievement**: ELO Rating 2000+ (Expert Level)

## 🏗️ Architecture

### Core Components

- **`baghchal.py`** - Game engine with advanced state evaluation
- **`training/mcts.py`** - Monte Carlo Tree Search implementation
- **`models/neural_network.py`** - Enhanced neural network architecture
- **`training/self_play.py`** - Self-play data generation system
- **`training/training_loop.py`** - Comprehensive training pipeline
- **`models/utils.py`** - Performance evaluation and ELO calculation

### Neural Network Architecture

- **Input**: 5×5×5 tensor (tiger layer, goat layer, empty layer, player turn, goats remaining)
- **Architecture**: 8 residual blocks with 512 filters, batch normalization, and dropout
- **Outputs**: 
  - Policy head: 65 actions (25 placements + 40 movements)
  - Value head: Game state evaluation (-1 to +1)
- **Features**: Action masking, L2 regularization, separate optimizers

### MCTS Implementation

- **Simulations**: 200 per move (configurable)
- **Exploration**: UCB1 with neural network prior scaling
- **Action Selection**: Temperature-controlled exploration during training
- **State Management**: Temporary game states for safe simulation

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Baghchal
   ```

2. **Create an environment** (use one of):
   - **venv**: `python -m venv .venv` then activate (e.g. `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux/Mac).
   - **conda**: `conda env create -f environment.yml` then `conda activate baghchal`.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU** (recommended for training)
   ```bash
   python test_gpu.py
   ```

5. **Verify installation**
   ```bash
   python tests/test_system.py
   ```

### Resuming training and inference

- **Resume from checkpoint**: `python main.py --resume models/bagh_chal_model_iter_5.keras`
- **Config**: Edit `config.yaml` for iterations, batch size, paths. Override with env: `BAGHCHAL_SAVE_DIR`, `BAGHCHAL_MODEL_PATH`.
- **Run inference** (after training a model): `python run_inference.py models/final_bagh_chal_model.keras --games 3`

### Run API locally (play in browser)

1. Train and save a model (e.g. `models/final_bagh_chal_model.keras`).
2. From the project root:
   ```bash
   set BAGHCHAL_MODEL_PATH=models\final_bagh_chal_model.keras
   uvicorn api.app:app --reload --port 8080
   ```
   (On Linux/Mac use `export BAGHCHAL_MODEL_PATH=...` and forward slash.)
3. Open http://localhost:8080 and click **New game**. You play Goats; the AI plays Tigers.

### Deploy to cloud (Railway, Render, Fly.io)

1. **Build**: Ensure a trained model is available. Either copy it into `models/` before building, or mount it at runtime.
2. **Docker**: From project root:
   ```bash
   docker build -t baghchal .
   docker run -p 8080:8080 -v "%cd%\models:/app/models" baghchal
   ```
   (Use `$(pwd)/models:/app/models` on Linux/Mac.) Set `MODEL_PATH=/app/models/final_bagh_chal_model.keras` if your file name differs.
3. **Platform**: Deploy the container to Railway, Render, or Fly.io using their Docker workflow. Set the port to **8080** and the start command to the default (uvicorn). On free tiers the app may spin down when idle; the first request after idle may be slow while the model loads.

## 🎮 Game Rules

### Board Setup
- 5×5 grid with tigers at corners
- Tigers start at positions (0,0), (0,4), (4,0), (4,4)

### Game Phases

#### Phase 1: Goat Placement (First 20 moves)
- Goats are placed one by one on empty cells
- Goal: Strategic positioning for blocking tigers

#### Phase 2: Movement
- Tigers and goats move alternately
- **Tiger Movement**: Adjacent cells or jump captures
- **Goat Movement**: Adjacent cells only
- **Capture**: Tigers can jump over goats to capture them

### Victory Conditions
- **Tiger Wins**: Capture 5 goats
- **Goat Wins**: Block all tigers from moving
- **Draw**: Repetitive game state

## 🧠 AI Training

### Training Pipeline

1. **Self-Play Generation**
   - Generate games using current model + MCTS
   - Temperature-controlled exploration
   - Quality metrics tracking

2. **Neural Network Training**
   - Policy and value head optimization
   - Learning rate scheduling
   - Regularization and dropout

3. **Performance Evaluation**
   - ELO rating calculation
   - Win rate analysis
   - Model comparison

### Training Parameters

```python
# Default training configuration
num_iterations = 20
games_per_iteration = 20
batch_size = 64
learning_rate = 0.001
num_simulations = 100
```

### Monitoring

- **Real-time metrics**: Loss, accuracy, win rates
- **Performance plots**: ELO progression, training curves
- **Model checkpoints**: Automatic saving every iteration

## 📊 Performance Evaluation

### ELO Rating System

- **Beginner**: 0-1200
- **Intermediate**: 1200-1600
- **Advanced**: 1600-2000
- **Expert**: 2000-2400
- **Master**: 2400+

### Evaluation Methods

1. **Random Play**: Baseline performance assessment
2. **Self-Play**: Internal consistency testing
3. **Baseline Comparison**: Model vs. previous versions
4. **Game Quality Analysis**: Training data assessment

## 🔧 Usage

### Basic Game Play

```python
from baghchal import BaghChalGame

# Initialize game
game = BaghChalGame()

# Place a goat
game.place_goat((2, 2))

# Make a move
game.make_move((0, 0), (1, 1))

# Check game status
status = game.check_victory_conditions()
```

### Training a Model

```python
from main import main

# Start training
main()
```

### Model Evaluation

```python
from models.utils import PerformanceEvaluator

evaluator = PerformanceEvaluator()
results = evaluator.evaluate_model_against_random(model, num_games=50)
print(f"Estimated ELO: {results['estimated_elo']:.0f}")
```

## 📁 Project Structure

```
Baghchal/
├── baghchal.py              # Main game engine
├── main.py                  # Training entry point
├── run_inference.py         # Inference script (load model, run games)
├── config.yaml              # Training and API config
├── config_loader.py         # Config loader with env overrides
├── game_actions.py         # Shared action encoding (0–64)
├── requirements.txt         # Dependencies
├── Dockerfile               # Container for API + frontend
├── api/                     # Web play vs AI
│   ├── app.py              # FastAPI app (new game, move, AI)
│   └── static/index.html   # Single-page board UI
├── models/                  # Neural network
│   ├── neural_network.py   # Architecture, load/save .keras
│   └── utils.py            # Performance evaluation
├── training/                # MCTS, self-play, training loop
│   ├── mcts.py             # Batched MCTS
│   ├── self_play.py        # Sequential and parallel self-play
│   └── training_loop.py    # Training pipeline, checkpoint manifest
├── tests/
└── notebooks/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_system.py

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Speed and memory testing
- **Game Logic Tests**: Rule validation testing

## 📈 Performance Optimization

### Training Improvements

1. **Hyperparameter Tuning**
   - Learning rate scheduling
   - Batch size optimization
   - Network architecture refinement

2. **Data Quality**
   - Self-play diversity enhancement
   - Action exploration strategies
   - Game length optimization

3. **Computational Efficiency**
   - GPU utilization optimization
   - Memory management
   - Parallel processing

### ELO 2000+ Strategy

1. **Extensive Training**: 100+ iterations with 50+ games each
2. **Quality Data**: High-quality self-play with diverse strategies
3. **Regular Evaluation**: Continuous performance monitoring
4. **Iterative Improvement**: Model refinement based on results

## 🤝 Contributing

### Development Guidelines

1. **Code Style**: Follow PEP 8 standards
2. **Documentation**: Comprehensive docstrings and comments
3. **Testing**: Maintain >90% test coverage
4. **Performance**: Optimize for speed and memory

### Areas for Contribution

- **Algorithm Improvements**: MCTS enhancements, evaluation functions
- **Neural Network**: Architecture optimization, training strategies
- **Game Logic**: Rule validation, edge case handling
- **Performance**: Speed optimization, memory management

## 📚 Research References

- **AlphaZero Paper**: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- **MCTS**: Monte Carlo Tree Search fundamentals
- **Deep Learning**: Neural network architectures for games
- **Game Theory**: Strategic game analysis

## 🏆 Achievements

- **Current Status**: AlphaZero implementation for Baghchal
- **Target**: ELO 2000+ (Expert Level)
- **Architecture**: State-of-the-art neural network design
- **Training**: Comprehensive self-play pipeline

## 📞 Support

### Issues and Questions

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check this README and code comments
- **Community**: Join discussions and share improvements

### Getting Help

1. Check the documentation and examples
2. Review existing issues and solutions
3. Create detailed issue reports with reproduction steps
4. Provide system information and error logs

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **AlphaZero Team**: Original research and algorithm
- **Open Source Community**: Libraries and tools
- **Game Enthusiasts**: Baghchal rules and strategy insights

---

**Goal**: Achieve ELO 2000+ and demonstrate the power of AlphaZero for traditional board games.

**Status**: Active development with continuous improvements.

**Next Milestone**: Complete training pipeline and achieve ELO 1500+.
