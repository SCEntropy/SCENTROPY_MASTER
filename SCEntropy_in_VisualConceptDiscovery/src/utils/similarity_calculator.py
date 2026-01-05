"""
Similarity calculation module for image classification experiments.
This module provides functionality to calculate various similarity metrics between features.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


class SimilarityCalculator:
    """Class for calculating similarity between features."""
    
    @staticmethod
    def calculate_cosine_similarity_matrix(avg_features_by_class: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Calculate cosine similarity matrix between class average features.
        
        Args:
            avg_features_by_class: Dictionary mapping class labels to average features
            
        Returns:
            Cosine similarity matrix
        """
        n_classes = len(avg_features_by_class)
        cosine_similarity_matrix = np.zeros((n_classes, n_classes))
        
        for i in range(n_classes):
            for j in range(n_classes):
                # Calculate cosine similarity between class i and class j
                feature_i = torch.tensor(avg_features_by_class[i])
                feature_j = torch.tensor(avg_features_by_class[j])
                
                # Calculate cosine similarity
                cosine_sim = F.cosine_similarity(feature_i, feature_j, dim=0).item()
                cosine_similarity_matrix[i, j] = cosine_sim
        
        return cosine_similarity_matrix
    
    @staticmethod
    def calculate_euclidean_distance_matrix(avg_features_by_class: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Calculate Euclidean distance matrix between class average features.
        
        Args:
            avg_features_by_class: Dictionary mapping class labels to average features
            
        Returns:
            Euclidean distance matrix
        """
        n_classes = len(avg_features_by_class)
        distance_matrix = np.zeros((n_classes, n_classes))
        
        for i in range(n_classes):
            for j in range(n_classes):
                # Calculate Euclidean distance between class i and class j
                diff = avg_features_by_class[i] - avg_features_by_class[j]
                distance = np.linalg.norm(diff)  # Euclidean distance
                distance_matrix[i, j] = distance
        
        return distance_matrix
    
    @staticmethod
    def save_similarity_matrix(similarity_matrix: np.ndarray, 
                              class_names: List[str], 
                              save_path: str, 
                              matrix_type: str = "cosine"):
        """
        Save similarity matrix to CSV file.
        
        Args:
            similarity_matrix: Similarity matrix to save
            class_names: Names of the classes
            save_path: Path to save the CSV file
            matrix_type: Type of matrix ('cosine' or 'euclidean')
        """
        # Create DataFrame with class names as both index and columns
        df = pd.DataFrame(similarity_matrix, 
                         index=[f"Class {i} ({name})" for i, name in enumerate(class_names)], 
                         columns=[f"Class {i} ({name})" for i, name in enumerate(class_names)])
        
        # Save to CSV
        df.to_csv(save_path)
        print(f"{matrix_type.capitalize()} similarity matrix saved to {save_path}")