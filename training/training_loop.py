import random
import numpy as np
import os
import json
from datetime import datetime
from training.self_play import generate_self_play_data
from models.neural_network import create_bagh_chal_model, get_training_callbacks
from keras._tf_keras.keras.utils import Sequence
from keras._tf_keras.keras.models import load_model
import matplotlib.pyplot as plt


class ReplayBuffer:
    """
    An enhanced replay buffer with prioritized sampling and better memory management.
    """
    def __init__(self, max_size=50000):
        self.buffer = []
        self.max_size = max_size
        self.priorities = []
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent

    def add(self, data):
        """Add new data with uniform priority"""
        self.buffer.extend(data)
        new_priorities = [1.0] * len(data)  # Uniform priority for new data
        self.priorities.extend(new_priorities)
        
        # Maintain buffer size
        if len(self.buffer) > self.max_size:
            excess = len(self.buffer) - self.max_size
            self.buffer = self.buffer[excess:]
            self.priorities = self.priorities[excess:]

    def sample(self, batch_size):
        """Sample data with prioritized sampling"""
        if len(self.buffer) == 0:
            return []
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return [self.buffer[i] for i in indices], weights

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled data"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority


class SelfPlayDataset(Sequence):
    """
    Enhanced dataset class with weighted sampling and better data handling.
    """
    def __init__(self, data, batch_size, weights=None):
        self.data = data
        self.batch_size = batch_size
        self.weights = weights if weights is not None else np.ones(len(data))

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_weights = self.weights[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        states, actions, values, masks = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        values = np.array(values)
        masks = np.array(masks)
        
        return [states, masks], {"policy": actions, "value": values}


class TrainingMetrics:
    """Track and visualize training progress"""
    
    def __init__(self):
        self.history = {
            'policy_loss': [], 'value_loss': [], 'total_loss': [],
            'policy_accuracy': [], 'value_mae': [],
            'self_play_win_rate': [], 'iteration': []
        }
    
    def update(self, metrics, iteration):
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        self.history['iteration'].append(iteration)
    
    def plot_training_progress(self, save_path="training_progress.png"):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(self.history['iteration'], self.history['policy_loss'], label='Policy Loss')
        axes[0, 0].plot(self.history['iteration'], self.history['value_loss'], label='Value Loss')
        axes[0, 0].plot(self.history['iteration'], self.history['total_loss'], label='Total Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        
        # Accuracy plots
        axes[0, 1].plot(self.history['iteration'], self.history['policy_accuracy'], label='Policy Accuracy')
        axes[0, 1].set_title('Policy Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Accuracy')
        
        # Value MAE
        axes[1, 0].plot(self.history['iteration'], self.history['value_mae'], label='Value MAE')
        axes[1, 0].set_title('Value Mean Absolute Error')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('MAE')
        
        # Win rate
        axes[1, 1].plot(self.history['iteration'], self.history['self_play_win_rate'], label='Tiger Win Rate')
        axes[1, 1].set_title('Self-Play Tiger Win Rate')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Win Rate')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics(self, save_path="training_metrics.json"):
        """Save metrics to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def train_bagh_chal_model(game, model, num_iterations=20, games_per_iteration=20, 
                          batch_size=64, save_dir="models", learning_rate=0.001):
    """
    Enhanced training function with better hyperparameters and monitoring.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize replay buffer and metrics
    replay_buffer = ReplayBuffer(max_size=100000)
    metrics = TrainingMetrics()
    
    # Training callbacks
    callbacks = get_training_callbacks(patience=5, factor=0.7, min_lr=1e-6)
    
    # Learning rate schedule
    initial_lr = learning_rate
    
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        
        # Adjust learning rate based on iteration
        current_lr = initial_lr * (0.9 ** (iteration // 5))
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Generate self-play data
        print("Generating self-play data...")
        self_play_data = generate_self_play_data(
            game, model, 
            num_games=games_per_iteration,
            num_simulations=100,  # Increased simulations
            temperature=1.0 if iteration < 5 else 0.5  # Reduce temperature over time
        )
        
        # Calculate win rate
        tiger_wins = sum(1 for _, _, value, _ in self_play_data if value > 0)
        win_rate = tiger_wins / len(self_play_data) if self_play_data else 0.5
        print(f"Tiger win rate: {win_rate:.3f}")
        
        # Add to replay buffer
        replay_buffer.add(self_play_data)
        
        # Sample training data
        if len(replay_buffer.buffer) >= batch_size:
            training_data, weights = replay_buffer.sample(min(len(replay_buffer.buffer), batch_size * 4))
            
            # Create dataset
            dataset = SelfPlayDataset(training_data, batch_size, weights)
            
            # Train model
            print("Training the model...")
            history = model.fit(
                dataset, 
                epochs=3,  # Increased epochs per iteration
                callbacks=callbacks,
                verbose=1
            )
            
            # Update metrics
            metrics.update({
                'policy_loss': history.history['policy_loss'][-1],
                'value_loss': history.history['value_loss'][-1],
                'total_loss': history.history['loss'][-1],
                'policy_accuracy': history.history['policy_accuracy'][-1],
                'value_mae': history.history['value_mae'][-1],
                'self_play_win_rate': win_rate
            }, iteration)
            
            print(f"Policy Loss: {history.history['policy_loss'][-1]:.4f}")
            print(f"Value Loss: {history.history['value_loss'][-1]:.4f}")
            print(f"Policy Accuracy: {history.history['policy_accuracy'][-1]:.4f}")
        
        # Save the model
        model_path = os.path.join(save_dir, f"bagh_chal_model_iter_{iteration + 1}.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save metrics every 5 iterations
        if (iteration + 1) % 5 == 0:
            metrics.save_metrics(os.path.join(save_dir, f"metrics_iter_{iteration + 1}.json"))
            metrics.plot_training_progress(os.path.join(save_dir, f"progress_iter_{iteration + 1}.png"))
    
    # Final metrics save
    metrics.save_metrics(os.path.join(save_dir, "final_metrics.json"))
    metrics.plot_training_progress(os.path.join(save_dir, "final_progress.png"))
    
    return model, metrics


if __name__ == "__main__":
    from baghchal import BaghChalGame

    # Initialize game and model
    game = BaghChalGame()
    model = create_bagh_chal_model(action_space_size=65, learning_rate=0.001)

    # Train the model with enhanced parameters
    trained_model, metrics = train_bagh_chal_model(
        game, model, 
        num_iterations=20, 
        games_per_iteration=20, 
        batch_size=64,
        learning_rate=0.001
    )
