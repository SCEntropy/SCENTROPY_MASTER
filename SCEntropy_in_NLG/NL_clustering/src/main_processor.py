from typing import List
from .embedding_clustering import EmbeddingClustering


class MainProcessor:
    """
    Main processor class that performs sentence clustering based on semantic similarity
    """
    
    def __init__(self, entropy_threshold: float = 1.0,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        self.entropy_threshold = entropy_threshold
        self.embedding_cluster = EmbeddingClustering(model_name=embedding_model)

    def _track_clusters(self, round_results: List[List[List[int]]], initial_count: int) -> List[List[List[int]]]:
        """
        Track cluster information across rounds
        """
        label_map = {i: [i] for i in range(initial_count)}
        tracking_info = []

        for result in round_results:
            current_round_info = []
            new_label_map = {}
            
            for new_label, cluster in enumerate(result):
                original_labels = []
                for item in cluster:
                    original_labels.extend(label_map[item])
                
                current_round_info.append(original_labels)
                new_label_map[new_label] = original_labels
            
            tracking_info.append(current_round_info)
            label_map = new_label_map

        return tracking_info

    def process_sentences_from_file(self, sentences: List[str], output_dir: str = '.') -> List[List[List[int]]]:
        """
        Process pre-generated sentences directly from file (skip AI generation)
        Only performs embedding calculation and clustering
        
        Args:
            sentences: List of pre-generated sentences
            output_dir: Directory to save output files
            
        Returns:
            List of tracking results from clustering
        """
        # Display all loaded sentences with labels
        print("\nAll sentences and their label information:")
        for i, sentence in enumerate(sentences):
            print(f"Sentence {i+1} (Label: {i}): {sentence}")
        
        # Calculate embeddings
        embeddings = self.embedding_cluster.compute_embeddings(sentences)
        features = {i: embeddings[i].reshape(1, -1) for i in range(len(sentences))}
        
        # Perform agglomerative clustering with entropy
        clusters, original_labels, round_results, original_labels_rec = self.embedding_cluster.agglomerative_clustering_with_entropy(
            features, self.entropy_threshold
        )
        
        # Track cluster information
        tracked_clusters = self._track_clusters(round_results, len(sentences))
        
        # Final Safe Result Section
        print("\n=== Final Safe Result ===")
        
        print("\nAll sentences with labels:")
        for i, sentence in enumerate(sentences):
            print(f"Label {i}: {sentence}")
        
        print("\nClustering tracking information:")
        for idx, info in enumerate(tracked_clusters):
            print(f"Round {idx}: {info}")
        
        return tracked_clusters
