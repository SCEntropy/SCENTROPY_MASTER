"""
File handling module for image classification experiments.
This module provides functionality to save and load features and other data.
"""

import numpy as np
import os
from typing import Dict, List, Any


class FileHandler:
    """Class for handling file operations."""
    
    @staticmethod
    def save_features_by_class(features_by_class: Dict[int, List[np.ndarray]], 
                              save_path: str) -> str:
        """
        Save features by class to a .npz file.
        
        Args:
            features_by_class: Dictionary mapping class labels to feature arrays
            save_path: Path to save the .npz file
            
        Returns:
            Path to the saved file
        """
        # Convert keys to strings to satisfy np.savez_compressed requirements
        features_by_class_str_keys = {str(k): v for k, v in features_by_class.items()}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save features
        np.savez_compressed(save_path, **features_by_class_str_keys)
        print(f"Features by class saved to {save_path}")
        
        return save_path
    
    @staticmethod
    def load_features_by_class(load_path: str) -> Dict[str, np.ndarray]:
        """
        Load features by class from a .npz file.
        
        Args:
            load_path: Path to the .npz file
            
        Returns:
            Dictionary mapping class labels to feature arrays
        """
        features = np.load(load_path)
        return features
    
    @staticmethod
    def create_results_directory(dataset_name: str) -> str:
        """
        Create a results directory for the specified dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to the created results directory
        """
        results_dir = os.path.join("results", dataset_name.lower())
        os.makedirs(results_dir, exist_ok=True)
        return results_dir