"""
Feature extraction module for image classification experiments.
This module provides functionality to extract features from pre-trained models.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple


class FeatureExtractor:
    """Class for extracting features from pre-trained models."""
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 10, 
                 pretrained: bool = True, device: str = None):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name of the model to use (currently only resnet50 supported)
            num_classes: Number of output classes for the final layer
            pretrained: Whether to use pretrained weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Initialize model
        if model_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Replace the final layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)
        
        # Remove the final classification layer for feature extraction
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
    def extract_features(self, data_loader: DataLoader) -> Dict[int, List[np.ndarray]]:
        """
        Extract features from the given data loader.
        
        Args:
            data_loader: DataLoader containing the data to extract features from
            
        Returns:
            Dictionary mapping class labels to lists of extracted features
        """
        features_by_class = {i: [] for i in range(self.num_classes)}
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Extracting features"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features
                features = self.feature_extractor(images).squeeze(-1).squeeze(-1)  # (batch_size, 2048)
                
                # Store features by class
                for feature, label in zip(features, labels):
                    features_by_class[label.item()].append(feature.cpu().numpy())
        
        return features_by_class
    
    def get_average_features(self, features_by_class: Dict[int, List[np.ndarray]]) -> Dict[int, np.ndarray]:
        """
        Calculate average features for each class.
        
        Args:
            features_by_class: Dictionary mapping class labels to lists of features
            
        Returns:
            Dictionary mapping class labels to average features
        """
        avg_features_by_class = {}
        for class_label, features in features_by_class.items():
            avg_features_by_class[class_label] = np.mean(features, axis=0)
        return avg_features_by_class