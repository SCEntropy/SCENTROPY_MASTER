#!/usr/bin/env python3
"""
Main script to run the complete SCEntropy in Visual Concept Discovery experiment pipeline.
This script orchestrates the entire process from data loading to clustering.
"""

import argparse
import os
import sys
import numpy as np
from typing import Dict, List

# Add project root to path to import config
sys.path.append(os.path.dirname(__file__))  # Add scripts dir
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add project root

from config import DEFAULT_CONFIG, DATASET_CONFIGS

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.dataset_loader import DatasetLoader
from models.model_trainer import ModelTrainer
from models.feature_extractor import FeatureExtractor
from clustering.entropy_clustering import EntropyBasedClustering
from utils.similarity_calculator import SimilarityCalculator
from utils.file_handler import FileHandler
from visualization.plots import Plotter
from torchvision.models import resnet50
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description='Run SCEntropy experiment')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['fashionmnist', 'cifar10', 'cifar100'],
                        help='Dataset to use for the experiment')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--entropy_threshold', type=float, default=None,
                        help='Threshold for entropy-based clustering (default values: 0.4 for fashionmnist/cifar10, 6.0 for cifar100)')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='Directory containing the datasets')
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='Directory for saving results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run the model on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set default entropy threshold based on dataset if not provided
    if args.entropy_threshold is None:
        args.entropy_threshold = DATASET_CONFIGS[args.dataset]['entropy_threshold']
    
    print(f"Starting SCEntropy experiment for {args.dataset}")
    print(f"Parameters: epochs={args.epochs}, threshold={args.entropy_threshold}")
    
    # Create results directory for this dataset
    dataset_results_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # 1. Load dataset
    print("1. Loading dataset...")
    dataset_loader = DatasetLoader(args.dataset, args.data_dir)
    train_loader, test_loader = dataset_loader.load_datasets(
        batch_size=args.batch_size
    )
    class_names = dataset_loader.get_class_names()
    
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
    
    # 5. Calculate average features
    print("5. Calculating average features...")
    avg_features_by_class = extractor.get_average_features(features_by_class)
    
    # 6. Calculate and save similarity matrices
    print("6. Calculating similarity matrices...")
    similarity_calc = SimilarityCalculator()
    
    # Calculate cosine similarity matrix
    cosine_matrix = similarity_calc.calculate_cosine_similarity_matrix(avg_features_by_class)
    cosine_save_path = os.path.join(dataset_results_dir, f"similarity_matrix_{dataset_loader.num_classes}.{dataset_loader.num_classes}.cossim.csv")
    similarity_calc.save_similarity_matrix(cosine_matrix, class_names, cosine_save_path, "cosine")
    
    # Calculate Euclidean distance matrix
    euclidean_matrix = similarity_calc.calculate_euclidean_distance_matrix(avg_features_by_class)
    euclidean_save_path = os.path.join(dataset_results_dir, f"similarity_matrix_{dataset_loader.num_classes}.{dataset_loader.num_classes}.avgFeature_distance.csv")
    similarity_calc.save_similarity_matrix(euclidean_matrix, class_names, euclidean_save_path, "euclidean")
    
    # 7. Perform entropy-based clustering
    print("7. Performing entropy-based clustering...")
    clustering = EntropyBasedClustering(entropy_threshold=args.entropy_threshold)
    clusters, original_labels, round_results, original_labels_rec = clustering.agglomerative_clustering_with_entropy(features_by_class)
    
    # 8. Track and display clustering results with original labels
    print("8. Tracking clustering results with original labels...")
    
    # Initialize a mapping to record each class's initial labels
    label_map = {i: [i] for i in range(dataset_loader.num_classes)}
    
    # Create a list to store tracking information for each round
    tracking_info = []
    
    # Iterate through each round of clustering
    for round_idx, result in enumerate(round_results):
        # Current round's tracking information
        current_round_info = []
        
        # New label mapping
        new_label_map = {}
        
        # Iterate through each cluster, building current round's tracking info
        for new_label, cluster in enumerate(result):
            # Find all initial labels in the current cluster
            original_labels_list = []
            for item in cluster:
                original_labels_list.extend(label_map[item])
            
            # Record the cluster with its original labels
            current_round_info.append(original_labels_list)
            
            # Update the new label mapping
            new_label_map[new_label] = original_labels_list
        
        # Save current round's tracking information
        tracking_info.append(current_round_info)
        
        # Update label mapping for next round
        label_map = new_label_map
    
    # Print tracking information to terminal
    print("\n" + "="*60)
    print("Clustering Round Results (with original labels):")
    print("="*60)
    for idx, info in enumerate(tracking_info, 1):
        print(f"Round {idx}: {info}")
    print("="*60 + "\n")
    
    # Save tracking information to file
    round_results_path = os.path.join(dataset_results_dir, "clustering_round_results.txt")
    with open(round_results_path, 'w') as f:
        f.write(f"Clustering Round Results for {args.dataset}\n")
        f.write(f"Entropy Threshold: {args.entropy_threshold}\n")
        f.write("="*60 + "\n")
        for idx, info in enumerate(tracking_info, 1):
            f.write(f"Round {idx}: {info}\n")
        f.write("="*60 + "\n")
    
    print(f"Clustering results saved to: {round_results_path}")
    
    # 9. Visualizations are disabled as per requirements
    print("9. Visualizations are disabled as per requirements")
    # The visualization code is available but not executed
    # To enable visualizations, uncomment the following code block
    
    print(f"\nExperiment completed! Results saved to {dataset_results_dir}")


if __name__ == "__main__":
    main()