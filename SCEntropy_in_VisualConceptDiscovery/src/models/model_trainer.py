"""
Model training module for image classification experiments.
This module provides functionality to train models on image datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple


class ModelTrainer:
    """Class for training image classification models."""
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Initialize the model trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, train_loader: DataLoader, num_epochs: int = 10, 
              learning_rate: float = 0.001, momentum: float = 0.9) -> list:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for the optimizer
            momentum: Momentum for SGD optimizer
            
        Returns:
            List of loss values for each epoch
        """
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        losses = []
        
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            losses.append(epoch_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        return losses
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the trained model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Tuple of (accuracy, average_loss)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss