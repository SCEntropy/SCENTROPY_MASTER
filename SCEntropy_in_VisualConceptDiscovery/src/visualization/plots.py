"""
Visualization module for image classification experiments.
This module provides functionality to create plots and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import List, Dict, Optional


class Plotter:
    """Class for creating various plots and visualizations."""
    
    @staticmethod
    def plot_dendrogram(centroid_array: np.ndarray, 
                       labels: List[str], 
                       method: str = 'ward',
                       title: str = "Hierarchical Clustering Dendrogram",
                       save_path: Optional[str] = None):
        """
        Plot a dendrogram for hierarchical clustering.
        
        Args:
            centroid_array: Array of centroids for each class
            labels: Labels for each class
            method: Linkage method ('ward', 'complete', 'average', etc.)
            title: Title for the plot
            save_path: Path to save the plot (optional)
        """
        # Perform hierarchical clustering
        Z = linkage(centroid_array, method=method, metric='euclidean')
        
        # Create the dendrogram plot
        plt.figure(figsize=(12, 8))
        dendrogram(Z, labels=labels, orientation='top', leaf_rotation=90)
        plt.title(title)
        plt.xlabel('Class Label')
        plt.ylabel('Distance')
        plt.tight_layout()
        
        if save_path:
            print(f"Dendrogram would be saved to {save_path} if visualization was enabled")
        
        # Skip showing the plot as per requirements
        # plt.show()
        plt.close()
    
    @staticmethod
    def plot_clustering_results(round_results: List[List[List[int]]], 
                               dataset_name: str = "Dataset"):
        """
        Plot the results of the clustering process across rounds.
        
        Args:
            round_results: List of clustering results for each round
            dataset_name: Name of the dataset for the title
        """
        num_rounds = len(round_results)
        
        if num_rounds == 0:
            print("No clustering results to plot.")
            return
        
        # Create a plot showing how clusters evolve over rounds
        plt.figure(figsize=(12, 6))
        
        for round_idx, result in enumerate(round_results):
            num_clusters = len(result)
            plt.scatter(round_idx, num_clusters, c='blue', s=50)
            plt.text(round_idx, num_clusters + 0.1, str(num_clusters), 
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Round')
        plt.ylabel('Number of Clusters')
        plt.title(f'Clustering Evolution Over Rounds - {dataset_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Skip showing the plot as per requirements
        # plt.show()
        plt.close()
    
    @staticmethod
    def plot_cluster_tracking_info(tracking_info: List[List[List[int]]], 
                                  dataset_name: str = "Dataset"):
        """
        Plot information about cluster tracking across rounds.
        
        Args:
            tracking_info: Tracking information for each round
            dataset_name: Name of the dataset for the title
        """
        if not tracking_info:
            print("No tracking info to plot.")
            return
        
        # Calculate the number of original classes in each cluster for each round
        cluster_sizes = []
        for round_idx, round_info in enumerate(tracking_info):
            sizes = [len(cluster) for cluster in round_info]
            cluster_sizes.append(sizes)
        
        # Create a plot showing cluster sizes over rounds
        plt.figure(figsize=(12, 6))
        
        for round_idx, sizes in enumerate(cluster_sizes):
            for i, size in enumerate(sizes):
                plt.scatter(round_idx, size, c='red', s=30, alpha=0.7)
                plt.text(round_idx, size + 0.1, str(size), 
                        ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Round')
        plt.ylabel('Cluster Size (Number of Original Classes)')
        plt.title(f'Cluster Size Evolution - {dataset_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Skip showing the plot as per requirements
        # plt.show()
        plt.close()