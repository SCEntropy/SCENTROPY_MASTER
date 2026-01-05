"""
Hierarchical Semantic Coherence (HSC) Metrics Module

This module implements custom evaluation metrics for clustering quality assessment,
specifically focusing on completeness metrics for semantic coherence evaluation.
"""
import numpy as np
from typing import List, Tuple, Dict, Any


class HSCMetrics:
    """
    Hierarchical Semantic Coherence (HSC) Metrics Calculator
    
    This class calculates custom completeness-based metrics for evaluating 
    the quality of clustering results, particularly in semantic coherence contexts.
    """
    
    def __init__(self, fine_to_coarse: Dict[int, int], coarse_classes: List[str]):
        """
        Initialize HSC Metrics calculator with semantic mapping.
        
        Args:
            fine_to_coarse: Mapping from fine-grained class indices to coarse-grained class indices
            coarse_classes: List of coarse-grained class names
        """
        self.fine_to_coarse = fine_to_coarse
        self.coarse_classes = coarse_classes
        self.num_coarse_classes = len(coarse_classes)
        
    def calculate_cluster_completeness(self, cluster_assignments: List[List[int]]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Calculate completeness scores for each cluster based on semantic coherence.
        
        Args:
            cluster_assignments: List of clusters, each containing fine-grained class indices
            
        Returns:
            Tuple of (completeness_scores, statistics_dict)
        """
        print("=" * 80)
        print("Custom Cluster Completeness Analysis")
        print("=" * 80)
        
        # Calculate total number of classes in each coarse category (in CIFAR-100 each has 5 classes)
        k_i = [5] * self.num_coarse_classes  # Each coarse category has 5 fine classes in CIFAR-100
        
        completeness_scores = []
        total_categories = 0
        
        print(f"Number of clusters: {len(cluster_assignments)}")
        
        # Process each cluster to calculate completeness
        for cluster_id, cluster in enumerate(cluster_assignments):
            n_j = len(cluster)  # Total number of fine classes in cluster j
            if n_j == 0:
                continue
                
            total_categories += n_j
            
            # Count number of fine classes for each coarse category in this cluster
            m_ij = [0] * self.num_coarse_classes  # Initialize counts for each coarse category
            
            for fine_idx in cluster:
                if fine_idx in self.fine_to_coarse:
                    coarse_idx = self.fine_to_coarse[fine_idx]
                    m_ij[coarse_idx] += 1
            
            # Calculate completeness for this cluster
            cluster_completeness = 0.0
            for i in range(self.num_coarse_classes):
                if m_ij[i] > 0:
                    cluster_completeness += (m_ij[i] / k_i[i]) * m_ij[i]
            cluster_completeness /= n_j if n_j > 0 else 1.0
            
            completeness_scores.append(cluster_completeness)
            
            # Print detailed information for each cluster
            print(f"\nCluster {cluster_id} (Size: {n_j}):")
            print(f"  Completeness Score: {cluster_completeness:.4f}")
            
            # Show distribution of classes in the cluster
            coarse_counts = {}
            for fine_idx in cluster:
                if fine_idx in self.fine_to_coarse:
                    coarse_idx = self.fine_to_coarse[fine_idx]
                    coarse_name = self.coarse_classes[coarse_idx]
                    coarse_counts[coarse_name] = coarse_counts.get(coarse_name, 0) + 1
            
            if coarse_counts:
                sorted_coarse = sorted(coarse_counts.items(), key=lambda x: x[1], reverse=True)
                for coarse, count in sorted_coarse:
                    percentage = (count / n_j) * 100 if n_j > 0 else 0
                    print(f"  {coarse}: {count} classes ({percentage:.1f}%)")
        
        # Calculate statistics
        if completeness_scores:
            mean_completeness = np.mean(completeness_scores)
            std_completeness = np.std(completeness_scores)
            min_completeness = np.min(completeness_scores)
            max_completeness = np.max(completeness_scores)
            perc_below_05 = (np.array(completeness_scores) < 0.5).sum() / len(completeness_scores) * 100
            
            stats = {
                'mean': mean_completeness,
                'std': std_completeness,
                'min': min_completeness,
                'max': max_completeness,
                'n': len(completeness_scores),
                'perc_below_05': perc_below_05,
                'completeness_scores': completeness_scores
            }
        else:
            stats = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'n': 0,
                'perc_below_05': 0.0,
                'completeness_scores': []
            }
        
        # Print overall results
        print(f"\n📊 Custom Completeness Metrics:")
        print(f"  Mean Completeness: {stats['mean']:.4f}")
        print(f"  Std Deviation: {stats['std']:.4f}")
        print(f"  Min Score: {stats['min']:.4f}")
        print(f"  Max Score: {stats['max']:.4f}")
        print(f"  N = {stats['n']}")
        
        # Metric explanation
        print(f"\n🎯 Metric Explanation:")
        print(f"  Completeness: Measure of how well each cluster covers its semantic categories")
        print(f"  Higher scores indicate better semantic coherence within clusters")
        
        # Evaluation
        print(f"\n📈 Evaluation:")
        if stats['mean'] >= 0.8:
            print(f"  ✅ Excellent - High clustering quality")
        elif stats['mean'] >= 0.6:
            print(f"  ⚠️  Good - Reasonable clustering quality")
        elif stats['mean'] >= 0.4:
            print(f"  ⚠️  Fair - Moderate clustering quality")
        else:
            print(f"  ❌ Poor - Clustering quality needs improvement")
        
        return completeness_scores, stats
    
    def compare_methods(self, method_results: Dict[str, List[List[int]]]) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple clustering methods using completeness metrics.
        
        Args:
            method_results: Dictionary mapping method names to cluster assignments
            
        Returns:
            Dictionary of statistics for each method
        """
        print("\n" + "="*80)
        print("COMPARISON OF CLUSTERING METHODS")
        print("="*80)
        
        all_stats = {}
        
        for method_name, cluster_assignments in method_results.items():
            print(f"\n--- {method_name} Method ---")
            completeness_scores, stats = self.calculate_cluster_completeness(cluster_assignments)
            all_stats[method_name] = stats
            all_stats[method_name]['completeness_scores'] = completeness_scores
        
        # Print comparison summary
        print(f"\n📋 COMPARISON SUMMARY:")
        print(f"{'Method':<20} {'N':<6} {'Mean':<8} {'Std':<8} {'<0.5%':<8}")
        print("-" * 60)
        for method_name, stats in all_stats.items():
            print(f"{method_name:<20} {stats['n']:<6} {stats['mean']:<8.3f} {stats['std']:<8.3f} {stats['perc_below_05']:<8.1f}%")
        
        return all_stats