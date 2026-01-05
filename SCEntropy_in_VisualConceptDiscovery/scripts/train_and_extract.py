#!/usr/bin/env python3
"""
Script to train a model and extract features.
"""

import argparse
import os
import sys

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.dataset_loader import DatasetLoader
from models.model_trainer import ModelTrainer
from models.feature_extractor import FeatureExtractor
from utils.file_handler import FileHandler
from torchvision.models import resnet50
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description='Train model and extract features')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['fashionmnist', 'cifar10', 'cifar100'],
                        help='Dataset to use for the experiment')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='Directory containing the datasets')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory for saving results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run the model on (cuda/cpu)')
    
    args = parser.parse_args()
    
    print(f"Training model and extracting features for {args.dataset}")
    
    # Create results directory for this dataset
    dataset_results_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # 1. Load dataset
    print("1. Loading dataset...")
    dataset_loader = DatasetLoader(args.dataset, args.data_dir)
    train_loader, test_loader = dataset_loader.load_datasets(
        batch_size=args.batch_size
    )
    
    # 2. Initialize and train model
    print("2. Training model...")
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, dataset_loader.num_classes)
    
    trainer = ModelTrainer(model, args.device)
    losses = trainer.train(train_loader, num_epochs=args.epochs)
    
    # 3. Extract features
    print("3. Extracting features...")
    extractor = FeatureExtractor(
        model_name='resnet50', 
        num_classes=dataset_loader.num_classes,
        pretrained=False  # Use the trained model
    )
    # Replace the model in extractor with the trained model
    extractor.model = trainer.model
    extractor.feature_extractor = nn.Sequential(*list(trainer.model.children())[:-1])
    extractor.feature_extractor = extractor.feature_extractor.to(extractor.device)
    extractor.feature_extractor.eval()
    
    features_by_class = extractor.extract_features(test_loader)
    
    # 4. Save features
    print("4. Saving features...")
    features_save_path = os.path.join(dataset_results_dir, "features_by_class.npz")
    FileHandler.save_features_by_class(features_by_class, features_save_path)
    
    print(f"Features extracted and saved to {features_save_path}")


if __name__ == "__main__":
    main()