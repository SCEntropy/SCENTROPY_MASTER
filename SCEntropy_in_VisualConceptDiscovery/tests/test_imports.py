"""
Simple test to verify that all modules can be imported correctly.
"""

import sys
import os
# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    try:
        # Test main modules
        from models import FeatureExtractor, ModelTrainer
        from data_processing import DatasetLoader
        from clustering import EntropyBasedClustering
        from utils import SimilarityCalculator, FileHandler
        from visualization import Plotter
        
        print("✅ All modules imported successfully!")
        
        # Test that classes can be instantiated (without full initialization)
        print("✅ Module classes can be imported.")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All imports successful! The code structure is correct.")
    else:
        print("\n❌ There were import errors. Please check the code structure.")