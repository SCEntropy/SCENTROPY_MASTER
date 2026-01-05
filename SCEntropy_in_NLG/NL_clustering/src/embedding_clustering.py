import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer


class EmbeddingClustering:
    """
    A class for handling sentence embeddings and clustering
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Compute sentence embeddings
        """
        return self.model.encode(sentences)
    
    def compute_centroids(self, features: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Calculate centroid for each class
        """
        centroids = {}
        for class_label in features:
            centroids[int(class_label)] = np.mean(features[class_label], axis=0).reshape(1, -1)
        return centroids

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors based on centroid
        """
        # Ensure input vectors are 1D arrays for scipy.spatial.distance.cosine
        vec1_flat = vec1.flatten()
        vec2_flat = vec2.flatten()
        return 0.5 * (1 - cosine(vec1_flat, vec2_flat))

    def calculate_complex_entropy(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate complex entropy
        """
        n = similarity_matrix.shape[0]
        complex_entropy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    for l in range(k + 1, n):
                        if (i, j) < (k, l):  # Ensure no duplicate calculations
                            S_ij = similarity_matrix[i, j]
                            S_kl = similarity_matrix[k, l]
                            complex_entropy -= np.log(1 - abs(S_ij - S_kl))
        return complex_entropy

    def update_labels(self, original_labels: List[int], new_cluster: List[int]) -> List[List[int]]:
        """
        Update labels to reflect clustering results
        """
        new_labels = []
        merged_labels_set = set(new_cluster)
        for label in original_labels:
            if label not in merged_labels_set:
                new_labels.append([label])
        new_labels.append(new_cluster)
        return new_labels

    def merge_centroids(self, original_centroids: Dict[int, np.ndarray], new_labels: List[List[int]]) -> Dict[int, np.ndarray]:
        """
        Recalculate centroids after merging
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
        Calculate similarity matrix
        """
        labels = list(centroids.keys())
        n = len(labels)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i, j] = self.cosine_similarity(centroids[labels[i]], centroids[labels[j]])
                else:
                    similarity_matrix[i, j] = 1.0
        return similarity_matrix, labels

    def relabel_centroids(self, original_centroids: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Re-sort labels each round
        """
        new_centroids = {}
        for new_label, old_label in enumerate(original_centroids.keys()):
            new_centroids[new_label] = original_centroids[old_label]
        return new_centroids

    def agglomerative_clustering_with_entropy(self, features: Dict[int, np.ndarray], entropy_threshold: float) -> Tuple[List[List[int]], Dict[int, List[int]], List[List[List[int]]], Dict[int, List[int]]]:
        """
        Perform agglomerative clustering based on complex entropy
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
            # If the complex entropy of remaining classes is below threshold, stop clustering
            end_loop = self.calculate_complex_entropy(similarity_matrix)
            # Output current complex entropy
            print(f"Round {round_count}: Complex Entropy = {end_loop}")
            if end_loop < entropy_threshold:
                print(f"Round {round_count}: Stopping Clustering. Complex Entropy = {end_loop}")
                break

            round_count += 1

            n_classes = len(centroids)
            clusters = [[i] for i in range(n_classes)]
            original_labels = {i: [i] for i in range(n_classes)}

            original_labels_1 = original_labels.copy()

            centroids = self.relabel_centroids(centroids)

            # Find the cluster pair (A, B) with maximum similarity
            max_s_value = -np.inf
            best_pair = None
            
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

            merged_label_index = len(original_labels)
            original_labels[merged_label_index] = new_labels
            
            merged = False
            while True:
                closest_category = None
                min_entropy = np.inf

                for C in range(len(clusters)):
                    if C != A and C != B:
                        temp_cluster = new_cluster + [C]
                        temp_entropy = self.calculate_complex_entropy(similarity_matrix[np.ix_(temp_cluster, temp_cluster)])
                        if temp_entropy < min_entropy:
                            min_entropy = temp_entropy
                            closest_category = C

                if closest_category is not None and min_entropy < entropy_threshold:
                    new_cluster.append(closest_category)
                    new_labels += original_labels[closest_category]
                    original_labels[merged_label_index] = new_labels
                    merged = True
                else:
                    break

            clusters = self.update_labels(original_labels_1, new_cluster)
            
            round_results.append(clusters)
            
            new_centroids = self.merge_centroids(centroids, clusters)
            centroids = new_centroids

            n_classes = len(centroids)
            similarity_matrix, labels = self.compute_similarity_matrix(centroids)

        return clusters, original_labels, round_results, original_labels_rec