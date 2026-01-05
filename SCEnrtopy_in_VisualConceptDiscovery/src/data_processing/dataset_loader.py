"""
Dataset loading module for image classification experiments.
This module provides functionality to load and preprocess image datasets.
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from typing import Tuple, Optional


class DatasetLoader:
    """Class for loading and preprocessing image datasets."""
    
    def __init__(self, dataset_name: str, data_dir: str):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_name: Name of the dataset ('fashionmnist', 'cifar10', or 'cifar100')
            data_dir: Directory containing the dataset
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.num_classes = self._get_num_classes()
        
    def _get_num_classes(self) -> int:
        """Get the number of classes for the specified dataset."""
        if self.dataset_name == 'fashionmnist':
            return 10
        elif self.dataset_name == 'cifar10':
            return 10
        elif self.dataset_name == 'cifar100':
            return 100
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get train and test transforms for the dataset."""
        if self.dataset_name == 'fashionmnist':
            # For FashionMNIST, we need to convert single channel to 3 channels
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Random horizontal flip (data augmentation)
                transforms.Resize((224, 224)),  # ResNet50 requires 224x224 input
                transforms.Grayscale(3),  # Convert to 3 channels
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
            
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # ResNet50 requires 224x224 input
                transforms.Grayscale(3),  # Convert to 3 channels
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            # For CIFAR-10 and CIFAR-100
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Random horizontal flip (data augmentation)
                transforms.Resize((224, 224)),  # ResNet50 requires 224x224 input
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
            
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # ResNet50 requires 224x224 input
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        
        return train_transform, test_transform
    
    def load_datasets(self, batch_size: int = 64, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        Load train and test datasets.
        
        Args:
            batch_size: Size of data batches
            num_workers: Number of worker processes for data loading
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        train_transform, test_transform = self._get_transforms()
        
        if self.dataset_name == 'fashionmnist':
            train_dataset = datasets.FashionMNIST(root=self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.FashionMNIST(root=self.data_dir, train=False, download=False, transform=test_transform)
        elif self.dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=False, transform=test_transform)
        elif self.dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(root=self.data_dir, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR100(root=self.data_dir, train=False, download=False, transform=test_transform)
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader
    
    def get_class_names(self) -> list:
        """
        Get the class names for the dataset.
        
        Returns:
            List of class names
        """
        if self.dataset_name == 'fashionmnist':
            return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        elif self.dataset_name == 'cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.dataset_name == 'cifar100':
            # CIFAR-100 has 100 classes, returning a placeholder
            # In practice, you'd use the actual CIFAR-100 class names
            return [f"class_{i}" for i in range(self.num_classes)]
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")