"""
Entropy-based clustering module.
This module implements clustering algorithms based on structural complexity entropy.
"""

import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple, Union
import torch.nn.functional as F
import torch


class EntropyBasedClustering:
    """Class for performing entropy-based clustering."""
    
    def __init__(self, entropy_threshold: float = 0.4):
        """
        Initialize the entropy-based clustering algorithm.
        
        Args:
            entropy_threshold: Threshold for stopping the clustering process
        """
        self.entropy_threshold = entropy_threshold
        
    def compute_centroids(self, features: Dict[str, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Compute centroids for each class.
        
        Args:
            features: Dictionary mapping class labels to feature arrays
            
        Returns:
            Dictionary mapping class labels to centroid vectors
        """
        centroids = {}
        for class_label in features:
            centroids[int(class_label)] = np.mean(features[class_label], axis=0)
        return centroids
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity value
        """
        return 1 - cosine(vec1, vec2)
    
    def calculate_complex_entropy(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate complex entropy based on the similarity matrix.
        
        Args:
            similarity_matrix: Matrix of similarities between classes
            
        Returns:
            Complex entropy value
        """
        n = similarity_matrix.shape[0]
        complex_entropy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    for l in range(k + 1, n):
                        if (i, j) < (k, l):  # Ensure no repeated calculations
                            S_ij = similarity_matrix[i, j]
                            S_kl = similarity_matrix[k, l]
                            complex_entropy -= np.log(1 - abs(S_ij - S_kl))
        return complex_entropy
    
    def update_labels(self, original_labels: Dict[int, List[int]], new_cluster: List[int]) -> List[List[int]]:
        """
        Update labels to reflect clustering results.
        
        Args:
            original_labels: Original label mappings
            new_cluster: New cluster to add
            
        Returns:
            Updated list of label groups
        """
        new_labels = []
        merged_labels_set = set(new_cluster)
        for label in original_labels:
            if label not in merged_labels_set:
                new_labels.append([label])
        new_labels.append(new_cluster)
        return new_labels
    
    def merge_centroids(self, original_centroids: Dict[int, np.ndarray], 
                       new_labels: List[List[int]]) -> Dict[int, np.ndarray]:
        """
        Recalculate centroids after merging.
        
        Args:
            original_centroids: Original centroids
            new_labels: New label groupings
            
        Returns:
            New centroids dictionary
        """
        new_centroids = {}
        for label_group in new_labels:
            if len(label_group) == 1:
                label = label_group[0]
                new_centroids[label] = original_centroids[label]
            else:
                combined_centroids = [original_centroids[label] for label in label_group]
                new_centroid = np.mean(combined_centroids, axis=0)
                new_centroids[label_group[0]] = new_centroid
        return new_centroids
    
    def compute_similarity_matrix(self, centroids: Dict[int, np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        """
        Compute similarity matrix based on centroids.
        
        Args:
            centroids: Dictionary mapping labels to centroids
            
        Returns:
            Similarity matrix and corresponding labels
        """
        labels = list(centroids.keys())
        n = len(labels)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i, j] = 1 - cosine(centroids[labels[i]], centroids[labels[j]])
                else:
                    similarity_matrix[i, j] = 1.0
        return similarity_matrix, labels
    
    def relabel_centroids(self, original_centroids: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Relabel centroids with new sequential indices.
        
        Args:
            original_centroids: Original centroids dictionary
            
        Returns:
            Relabeled centroids dictionary
        """
        new_centroids = {}
        for new_label, old_label in enumerate(original_centroids.keys()):
            new_centroids[new_label] = original_centroids[old_label]
        return new_centroids
    
    def agglomerative_clustering_with_entropy(self, features: Dict[str, np.ndarray]) -> Tuple[List[List[int]], Dict[int, List[int]], List[List[List[int]]], Dict[int, List[int]]]:
        """
        Perform agglomerative clustering based on complex entropy.
        
        Args:
            features: Dictionary mapping class labels to feature arrays
            
        Returns:
            Tuple of (final_clusters, original_labels, round_results, original_labels_rec)
        """
        centroids = self.compute_centroids(features)
        n_classes = len(centroids)
        clusters = [[i] for i in range(n_classes)]
        original_labels = {i: [i] for i in range(n_classes)}
        original_labels_rec = original_labels.copy()
        round_results = []
        round_results.append(clusters)
        
        similarity_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                similarity_matrix[i, j] = self.cosine_similarity(centroids[i], centroids[j])
                similarity_matrix[j, i] = similarity_matrix[i, j]

        round_count = 0

        while True:
            # Stop clustering if the complex entropy of remaining classes is below threshold
            end_loop = self.calculate_complex_entropy(similarity_matrix)
            if end_loop < self.entropy_threshold:
                break

            round_count += 1
            print(f"Round {round_count}: starting with {len(clusters)} clusters.")
            print(f"Clusters: {clusters}")
            print(f"Similarity matrix:\n{similarity_matrix}")

            n_classes = len(centroids)
            clusters = [[i] for i in range(n_classes)]
            original_labels = {i: [i] for i in range(n_classes)}

            original_labels_1 = original_labels.copy()

            print(f"Centroids before round {round_count}: {centroids}")
            centroids = self.relabel_centroids(centroids)

            # Find the pair of clusters with maximum similarity (A, B)
            max_s_value = -np.inf
            best_pair = None
            
            print(f"Number of clusters: {len(clusters)}")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if similarity_matrix[i, j] > max_s_value:
                        max_s_value = similarity_matrix[i, j]
                        best_pair = (i, j)

            if best_pair is None:
                break

            A, B = best_pair
            new_cluster = clusters[A] + clusters[B]
            new_labels = original_labels[A] + original_labels[B]
            print(f"Merged cluster {A} and {B} into new cluster {new_cluster} with labels {new_labels}")

            merged_label_index = len(original_labels)
            original_labels[merged_label_index] = new_labels
            
            merged = False

            added_categories = set()  # Record categories that have been added to new cluster

            while True:
                closest_category = None
                min_entropy = np.inf  # Initialize minimum entropy value

                # We need to use the current cluster indices for similarity matrix access
                for c in range(len(clusters)):
                    if c != A and c != B and c not in added_categories:  # Exclude already merged categories
                        # Calculate maximum similarity from C to AB (minimum distance)
                        # Use the current similarity matrix indices
                        c_to_ab_sim = max(similarity_matrix[c, A], similarity_matrix[c, B])
                        
                        # Calculate maximum similarity from C to other points
                        c_to_others_max_sim = -np.inf
                        for other_c in range(len(clusters)):
                            if other_c != A and other_c != B and other_c != c and other_c not in added_categories:
                                if similarity_matrix[c, other_c] > c_to_others_max_sim:
                                    c_to_others_max_sim = similarity_matrix[c, other_c]
                        
                        # Only consider adding C to AB if C's similarity to AB is >= C's similarity to other points
                        if c_to_ab_sim >= c_to_others_max_sim:
                            temp_cluster = new_cluster + [c]
                            temp_entropy = self.calculate_complex_entropy(similarity_matrix[np.ix_(temp_cluster, temp_cluster)])

                            if temp_entropy < min_entropy:
                                min_entropy = temp_entropy
                                closest_category = c

                if closest_category is not None and min_entropy < self.entropy_threshold:
                    # Ensure the added category is not duplicated
                    if closest_category not in added_categories:
                        new_cluster.append(closest_category)
                        new_labels += original_labels[closest_category]
                        added_categories.add(closest_category)  # Mark as merged
                        print(f"Added category {closest_category} to new cluster {new_cluster} with labels {new_labels}")

                    original_labels[merged_label_index] = new_labels
                    merged = True  # Mark as merged
                else:
                    break  # No suitable category to merge, exit loop


            clusters = self.update_labels(original_labels_1, new_cluster)
            
            print(f"Clusters after round {round_count}: {clusters}")
            round_results.append(clusters)
            
            new_centroids = self.merge_centroids(centroids, clusters)
            centroids = new_centroids
            print(f"Centroids after round {round_count}: {centroids}")

            n_classes = len(centroids)
            similarity_matrix, labels = self.compute_similarity_matrix(centroids)
            print(f"Similarity matrix after round {round_count}:\n{similarity_matrix}")

        print("\nAll rounds clustering results:")
        for idx, result in enumerate(round_results):
            print(f"Round {idx}: {result}")
        
        return clusters, original_labels, round_results, original_labels_rec