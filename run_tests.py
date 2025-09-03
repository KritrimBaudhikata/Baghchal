#!/usr/bin/env python3
"""
Test runner script for Baghchal AlphaZero system.
This script runs tests from the tests folder with proper import handling.
"""

import sys
import os
import subprocess

def run_tests():
    """Run all tests in the tests folder"""
    print("Running Baghchal AlphaZero Tests")
    print("=" * 40)
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(current_dir, "tests")
    
    # Check if tests directory exists
    if not os.path.exists(tests_dir):
        print("❌ Tests directory not found!")
        return False
    
    # Run the main test file
    test_file = os.path.join(tests_dir, "test_system.py")
    if os.path.exists(test_file):
        print(f"Running: {test_file}")
        try:
            # Run the test with proper Python path
            result = subprocess.run([
                sys.executable, test_file
            ], cwd=current_dir, capture_output=False)
            
            if result.returncode == 0:
                print("✅ All tests completed successfully!")
                return True
            else:
                print("❌ Tests failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    else:
        print("❌ Test file not found!")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
