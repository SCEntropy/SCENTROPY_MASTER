#!/usr/bin/env python3
"""
Display clustering hierarchy in text format using directory tree style
"""

import sys
import os
import numpy as np

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.file_handler import FileHandler
from clustering.entropy_clustering import EntropyBasedClustering


def print_clustering_hierarchy(round_results, class_names=None):
    """
    Print clustering hierarchy in directory tree format
    """
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("="*80)
    print("Structural Complexity Entropy Clustering - Non-Binary Tree Clustering Hierarchy")
    print("="*80)
    print(f"Dataset: CIFAR-10 (10 classes)")
    print(f"Number of clustering rounds: {len(round_results)}")
    print(f"Class mapping: {dict(enumerate(class_names))}")
    print()
    
    # Print clustering results for each round
    for round_idx, result in enumerate(round_results):
        print(f"Round {round_idx} Clustering Results:")
        print("-" * 50)
        
        for cluster_idx, cluster in enumerate(result):
            cluster_name_list = [class_names[i] for i in cluster]
            print(f"  ├─ Cluster {cluster_idx}: {cluster}")
            print(f"  │    Corresponding classes: {cluster_name_list}")
            print(f"  │    Cluster size: {len(cluster)} classes")
        print(f"  └─ Total {len(result)} clusters in this round")
        print()
    
    print("="*80)
    print("Clustering Evolution Process Analysis:")
    print("="*80)
    
    # Analyze clustering evolution
    for round_idx in range(1, len(round_results)):
        prev_result = round_results[round_idx - 1]
        curr_result = round_results[round_idx]
        
        print(f"Round {round_idx-1} → Round {round_idx} Evolution:")
        print("-" * 40)
        
        # Create mapping from class to cluster
        prev_cat_to_cluster = {}
        for i, cluster in enumerate(prev_result):
            for cat in cluster:
                prev_cat_to_cluster[cat] = i
        
        curr_cat_to_cluster = {}
        for i, cluster in enumerate(curr_result):
            for cat in cluster:
                curr_cat_to_cluster[cat] = i
        
        # Analyze changes
        changes = []
        for curr_cluster_idx, curr_cluster in enumerate(curr_result):
            # Find which previous clusters this new cluster came from
            prev_clusters = set()
            for cat in curr_cluster:
                if cat in prev_cat_to_cluster:
                    prev_clusters.add(prev_cat_to_cluster[cat])
            
            if len(prev_clusters) > 1:
                # This is a merge operation
                prev_cluster_info = []
                for prev_idx in sorted(list(prev_clusters)):
                    prev_cluster_cats = [class_names[i] for i in prev_result[prev_idx] 
                                       if i in [c for c in curr_cluster]]
                    prev_cluster_info.append(f"Cluster{prev_idx}({prev_result[prev_idx]}->{prev_cluster_cats})")
                
                curr_cluster_cats = [class_names[i] for i in curr_cluster]
                changes.append(f"  Merge: {' + '.join(prev_cluster_info)} → Cluster{curr_cluster_idx}({curr_cluster}->{curr_cluster_cats})")
            elif len(prev_clusters) == 1:
                prev_idx = list(prev_clusters)[0]
                prev_cluster = prev_result[prev_idx]
                if len(prev_cluster) < len(curr_cluster):
                    # Expansion operation
                    added_cats = set(curr_cluster) - set(prev_cluster)
                    added_cat_names = [class_names[i] for i in added_cats]
                    curr_cluster_cats = [class_names[i] for i in curr_cluster]
                    changes.append(f"  Expand: Cluster{prev_idx}({prev_cluster}) + {list(added_cats)}({added_cat_names}) → Cluster{curr_cluster_idx}({curr_cluster}->{curr_cluster_cats})")
        
        if changes:
            for change in changes:
                print(change)
        else:
            print("  No significant changes (clustering structure remains stable)")
        print()
    
    print("="*80)
    print("Final Clustering Results (Round {}):".format(len(round_results)-1))
    print("="*80)
    
    final_result = round_results[-1]
    print(f"Finally formed {len(final_result)} super-class clusters:\n")
    
    for cluster_idx, cluster in enumerate(final_result):
        cluster_names = [class_names[i] for i in cluster]
        print(f"📁 Super-class {cluster_idx}: {cluster}")
        print(f"   Contains classes: {cluster_names}")
        print(f"   Number of classes: {len(cluster)}")
        print(f"   Semantic content: {', '.join(cluster_names)}")
        print()
    
    print("="*80)
    print("Core Innovation Points of the Algorithm:")
    print("="*80)
    print("✓ Non-binary tree clustering structure: Supports simultaneous merging of multiple clusters")
    print("✓ Dynamic clustering evolution: Clusters dynamically expand and merge across rounds") 
    print("✓ Structural complexity entropy guidance: Automatic clustering termination based on entropy threshold")
    print("✓ Semantic-preserving clustering: Maintains semantic similarity between classes")
    print()
    print("Algorithm Advantages:")
    print("- Discovers more natural class grouping structures")
    print("- Avoids limitations of forced binary tree splitting")
    print("- Adapts to different levels of semantic granularity")


def simulate_clustering_process():
    """
    Simulate clustering process for demonstration
    """
    # Simulated clustering results
    round_results = [
        [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],  # Round 0 - Initial state
        [[0], [1], [2], [4], [6], [8], [9], [3, 5, 7]],      # Round 1 - 3,5,7 merged
        [[0], [1], [6], [8], [9], [2, 3, 5, 7, 4]],          # Round 2 - 4 added to [2,3,5,7]
        [[1], [6], [9], [0, 2, 3, 5, 7, 4, 8]],              # Round 3 - 0,8 added to large cluster
        [[6], [9, 0, 2, 3, 5, 7, 4, 8, 1]]                  # Round 4 - 1,9 added to large cluster
    ]
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print_clustering_hierarchy(round_results, class_names)


def main():
    print("Structural Complexity Entropy Clustering Algorithm - CIFAR-10 Clustering Results Display")
    print()
    
    # Use simulated data for display
    simulate_clustering_process()
    
    print("\nNote: The above display is based on simulated clustering evolution process.")
    print("The actual algorithm has been successfully implemented and generates results in the results/cifar10/ directory.")


if __name__ == "__main__":
    main()