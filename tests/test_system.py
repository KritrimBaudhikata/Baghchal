#!/usr/bin/env python3
"""
Test script to verify the enhanced Baghchal AlphaZero system.
"""

import sys
import os
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baghchal import BaghChalGame
from models.neural_network import create_bagh_chal_model
from models.utils import PerformanceEvaluator

def test_basic_functionality():
    """Test basic game and model functionality"""
    print("=== Testing Basic Functionality ===")
    
    # Test game
    game = BaghChalGame()
    print("✓ Game initialized successfully")
    
    # Test state serialization
    state = game.serialize_state_binary()
    print(f"✓ State serialization works: {state.shape}")
    
    # Test state deserialization
    game.deserialize_state_binary(state)
    print("✓ State deserialization works")
    
    # Test valid moves
    tiger_moves = game.get_valid_moves(1)
    goat_moves = game.get_valid_moves(2)
    print(f"✓ Valid moves: Tiger={len(tiger_moves)}, Goat={len(goat_moves)}")
    
    return True

def test_neural_network():
    """Test neural network creation and basic operations"""
    print("\n=== Testing Neural Network ===")
    
    try:
        model = create_bagh_chal_model(action_space_size=65, learning_rate=0.001)
        print("✓ Neural network created successfully")
        
        # Test model input/output
        test_state = np.random.random((1, 5, 5, 5))
        test_mask = np.ones((1, 65))
        
        policy, value = model.predict([test_state, test_mask], verbose=0)
        print(f"✓ Model prediction works: Policy={policy.shape}, Value={value.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def test_performance_evaluator():
    """Test performance evaluator functionality"""
    print("\n=== Testing Performance Evaluator ===")
    
    try:
        evaluator = PerformanceEvaluator(base_elo=1200)
        print("✓ Performance evaluator created successfully")
        
        # Test ELO calculation
        elo_change = evaluator.calculate_elo_change(1200, 1200, 1.0)
        print(f"✓ ELO calculation works: {elo_change:.2f}")
        
        # Test ELO update
        new_elo = evaluator.update_elo(1200, elo_change)
        print(f"✓ ELO update works: {new_elo:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Performance evaluator test failed: {e}")
        return False

def test_game_mechanics():
    """Test game mechanics and rules"""
    print("\n=== Testing Game Mechanics ===")
    
    try:
        game = BaghChalGame()
        
        # Test goat placement
        success = game.place_goat((2, 2))
        print(f"✓ Goat placement: {success}")
        
        # Test tiger movement
        if len(game.get_valid_moves(1)) > 0:
            print("✓ Tiger has valid moves")
        
        # Test victory conditions
        status = game.check_victory_conditions()
        print(f"✓ Game status check: {status}")
        
        return True
    except Exception as e:
        print(f"✗ Game mechanics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Baghchal AlphaZero System Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_neural_network,
        test_performance_evaluator,
        test_game_mechanics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for training.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
