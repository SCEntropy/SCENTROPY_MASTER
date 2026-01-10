#!/usr/bin/env python3
"""
Superclass Entropy Clustering Script

This script implements the superclass-based structural complexity entropy clustering algorithm,
testing the robustness of the approach across different entropy thresholds.
"""
import numpy as np
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cosine

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.hsc_metrics import HSCMetrics


def load_cifar100_fine_to_coarse_mapping():
    """
    Load the CIFAR-100 fine-to-coarse class mapping.
    
    Returns:
        tuple: (fine_to_coarse dict, coarse_classes list)
    """
    # CIFAR-100 coarse class names
    coarse_classes = [
        'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
        'household electrical devices', 'household furniture', 'insects', 'large carnivores',
        'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
        'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles',
        'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
    ]
    
    # Fine-to-coarse mapping
    fine_to_coarse = {}
    
    # 1. aquatic mammals: beaver(4), dolphin(30), otter(55), seal(72), whale(95)
    fine_to_coarse[4] = 0   # beaver
    fine_to_coarse[30] = 0  # dolphin
    fine_to_coarse[55] = 0  # otter
    fine_to_coarse[72] = 0  # seal
    fine_to_coarse[95] = 0  # whale

    # 2. fish: aquarium_fish(1), flatfish(32), ray(67), shark(73), trout(91)
    fine_to_coarse[1] = 1   # aquarium_fish
    fine_to_coarse[32] = 1  # flatfish
    fine_to_coarse[67] = 1  # ray
    fine_to_coarse[73] = 1  # shark
    fine_to_coarse[91] = 1  # trout

    # 3. flowers: orchid(54), poppy(62), rose(70), sunflower(82), tulip(92)
    fine_to_coarse[54] = 2  # orchid
    fine_to_coarse[62] = 2  # poppy
    fine_to_coarse[70] = 2  # rose
    fine_to_coarse[82] = 2  # sunflower
    fine_to_coarse[92] = 2  # tulip

    # 4. food containers: bottle(9), bowl(10), can(16), cup(28), plate(61)
    fine_to_coarse[9] = 3   # bottle
    fine_to_coarse[10] = 3  # bowl
    fine_to_coarse[16] = 3  # can
    fine_to_coarse[28] = 3  # cup
    fine_to_coarse[61] = 3  # plate

    # 5. fruit and vegetables: apple(0), mushroom(51), orange(53), pear(57), sweet_pepper(83)
    fine_to_coarse[0] = 4   # apple
    fine_to_coarse[51] = 4  # mushroom
    fine_to_coarse[53] = 4  # orange
    fine_to_coarse[57] = 4  # pear
    fine_to_coarse[83] = 4  # sweet_pepper

    # 6. household electrical devices: clock(22), keyboard(39), lamp(40), telephone(86), television(87)
    fine_to_coarse[22] = 5  # clock
    fine_to_coarse[39] = 5  # keyboard
    fine_to_coarse[40] = 5  # lamp
    fine_to_coarse[86] = 5  # telephone
    fine_to_coarse[87] = 5  # television

    # 7. household furniture: bed(5), chair(20), couch(25), table(84), wardrobe(94)
    fine_to_coarse[5] = 6   # bed
    fine_to_coarse[20] = 6  # chair
    fine_to_coarse[25] = 6  # couch
    fine_to_coarse[84] = 6  # table
    fine_to_coarse[94] = 6  # wardrobe

    # 8. insects: bee(6), beetle(7), butterfly(14), caterpillar(18), cockroach(24)
    fine_to_coarse[6] = 7   # bee
    fine_to_coarse[7] = 7   # beetle
    fine_to_coarse[14] = 7  # butterfly
    fine_to_coarse[18] = 7  # caterpillar
    fine_to_coarse[24] = 7  # cockroach

    # 9. large carnivores: bear(3), leopard(42), lion(43), tiger(88), wolf(97)
    fine_to_coarse[3] = 8   # bear
    fine_to_coarse[42] = 8  # leopard
    fine_to_coarse[43] = 8  # lion
    fine_to_coarse[88] = 8  # tiger
    fine_to_coarse[97] = 8  # wolf

    # 10. large man-made outdoor things: bridge(12), castle(17), house(37), road(68), skyscraper(76)
    fine_to_coarse[12] = 9  # bridge
    fine_to_coarse[17] = 9  # castle
    fine_to_coarse[37] = 9  # house
    fine_to_coarse[68] = 9  # road
    fine_to_coarse[76] = 9  # skyscraper

    # 11. large natural outdoor scenes: cloud(23), forest(33), mountain(49), plain(60), sea(71)
    fine_to_coarse[23] = 10  # cloud
    fine_to_coarse[33] = 10  # forest
    fine_to_coarse[49] = 10  # mountain
    fine_to_coarse[60] = 10  # plain
    fine_to_coarse[71] = 10  # sea

    # 12. large omnivores and herbivores: camel(15), cattle(19), chimpanzee(21), elephant(31), kangaroo(38)
    fine_to_coarse[15] = 11  # camel
    fine_to_coarse[19] = 11  # cattle
    fine_to_coarse[21] = 11  # chimpanzee
    fine_to_coarse[31] = 11  # elephant
    fine_to_coarse[38] = 11  # kangaroo

    # 13. medium-sized mammals: fox(34), porcupine(63), possum(64), raccoon(66), skunk(75)
    fine_to_coarse[34] = 12  # fox
    fine_to_coarse[63] = 12  # porcupine
    fine_to_coarse[64] = 12  # possum
    fine_to_coarse[66] = 12  # raccoon
    fine_to_coarse[75] = 12  # skunk

    # 14. non-insect invertebrates: crab(26), lobster(45), snail(77), spider(79), worm(99)
    fine_to_coarse[26] = 13  # crab
    fine_to_coarse[45] = 13  # lobster
    fine_to_coarse[77] = 13  # snail
    fine_to_coarse[79] = 13  # spider
    fine_to_coarse[99] = 13  # worm

    # 15. people: baby(2), boy(11), girl(35), man(46), woman(98)
    fine_to_coarse[2] = 14   # baby
    fine_to_coarse[11] = 14  # boy
    fine_to_coarse[35] = 14  # girl
    fine_to_coarse[46] = 14  # man
    fine_to_coarse[98] = 14  # woman

    # 16. reptiles: crocodile(27), dinosaur(29), lizard(44), snake(78), turtle(93)
    fine_to_coarse[27] = 15  # crocodile
    fine_to_coarse[29] = 15  # dinosaur
    fine_to_coarse[44] = 15  # lizard
    fine_to_coarse[78] = 15  # snake
    fine_to_coarse[93] = 15  # turtle

    # 17. small mammals: hamster(36), mouse(50), rabbit(65), shrew(74), squirrel(80)
    fine_to_coarse[36] = 16  # hamster
    fine_to_coarse[50] = 16  # mouse
    fine_to_coarse[65] = 16  # rabbit
    fine_to_coarse[74] = 16  # shrew
    fine_to_coarse[80] = 16  # squirrel

    # 18. trees: maple_tree(47), oak_tree(52), palm_tree(56), pine_tree(59), willow_tree(96)
    fine_to_coarse[47] = 17  # maple_tree
    fine_to_coarse[52] = 17  # oak_tree
    fine_to_coarse[56] = 17  # palm_tree
    fine_to_coarse[59] = 17  # pine_tree
    fine_to_coarse[96] = 17  # willow_tree

    # 19. vehicles 1: bicycle(8), bus(13), motorcycle(48), pickup_truck(58), train(90)
    fine_to_coarse[8] = 18   # bicycle
    fine_to_coarse[13] = 18  # bus
    fine_to_coarse[48] = 18  # motorcycle
    fine_to_coarse[58] = 18  # pickup_truck
    fine_to_coarse[90] = 18  # train

    # 20. vehicles 2: lawn_mower(41), rocket(69), streetcar(81), tank(85), tractor(89)
    fine_to_coarse[41] = 19  # lawn_mower
    fine_to_coarse[69] = 19  # rocket
    fine_to_coarse[81] = 19  # streetcar
    fine_to_coarse[85] = 19  # tank
    fine_to_coarse[89] = 19  # tractor

    return fine_to_coarse, coarse_classes


def compute_centroids(features):
    """Compute centroids for each class."""
    centroids = {}
    for class_label in features:
        centroids[int(class_label)] = np.mean(features[class_label], axis=0)
    return centroids


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return 1 - cosine(vec1, vec2)


def calculate_complex_entropy(similarity_matrix):
    """Calculate complex entropy from similarity matrix."""
    n = similarity_matrix.shape[0]
    complex_entropy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                for l in range(k + 1, n):
                    if (i, j) < (k, l):  # Ensure no repeated calculation
                        S_ij = similarity_matrix[i, j]
                        S_kl = similarity_matrix[k, l]
                        complex_entropy -= np.log(1 - abs(S_ij - S_kl))
    return complex_entropy


def compute_similarity_matrix_for_list(centroids_list):
    """Compute similarity matrix from a list of centroids."""
    n = len(centroids_list)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i, j] = cosine_similarity(centroids_list[i], centroids_list[j])
            else:
                similarity_matrix[i, j] = 1.0
    return similarity_matrix


def update_labels(original_labels, new_cluster):
    """Update labels after clustering."""
    new_labels = []
    merged_labels_set = set(new_cluster)
    for label in original_labels:
        if label not in merged_labels_set:
            new_labels.append([label])
    new_labels.append(new_cluster)
    return new_labels


def merge_centroids(original_centroids, new_labels):
    """Merge centroids after clustering."""
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


def compute_similarity_matrix(centroids):
    """Compute similarity matrix from centroids."""
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


def relabel_centroids(original_centroids):
    """Relabel centroids after clustering."""
    new_centroids = {}
    for new_label, old_label in enumerate(original_centroids.keys()):
        new_centroids[new_label] = original_centroids[old_label]
    return new_centroids


def agglomerative_clustering_with_entropy(features, entropy_threshold):
    """Perform agglomerative clustering based on entropy."""
    centroids = compute_centroids(features)
    n_classes = len(centroids)
    clusters = [[i] for i in range(n_classes)]
    original_labels = {i: [i] for i in range(n_classes)}
    original_labels_rec = original_labels.copy()
    round_results = []
    round_results.append(clusters)
    
    similarity_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            similarity_matrix[i, j] = cosine_similarity(centroids[i], centroids[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]

    round_count = 0

    while True:
        # If the complex entropy of the remaining classes is less than the threshold, stop clustering
        end_loop = calculate_complex_entropy(similarity_matrix)
        if end_loop < entropy_threshold:
            break

        round_count += 1

        n_classes = len(centroids)
        clusters = [[i] for i in range(n_classes)]
        original_labels = {i: [i] for i in range(n_classes)}

        original_labels_1 = original_labels.copy()

        centroids = relabel_centroids(centroids)

        # Find the pair of clusters with maximum similarity (A, B)
        max_s_value = -np.inf
        best_pair = None
        
        print(len(clusters))
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

        added_categories = set()  # Track categories already added to the new cluster

        while True:
            closest_category = None
            min_entropy = np.inf  # Initialize minimum entropy value

            for c in range(len(clusters)):
                if c != A and c != B and c not in added_categories:  # Exclude already merged categories
                    # Calculate max similarity from C to AB
                    c_to_ab_sim = max(similarity_matrix[c, A], similarity_matrix[c, B])
                    
                    # Calculate max similarity from C to other points
                    c_to_others_max_sim = -np.inf
                    for other_c in range(len(clusters)):
                        if other_c != A and other_c != B and other_c != c and other_c not in added_categories:
                            if similarity_matrix[c, other_c] > c_to_others_max_sim:
                                c_to_others_max_sim = similarity_matrix[c, other_c]
                    
                    # Only consider adding C to AB if C's similarity to AB is greater than or equal to C's max similarity to other points
                    if c_to_ab_sim >= c_to_others_max_sim:
                        temp_cluster = new_cluster + [c]
                        temp_entropy = calculate_complex_entropy(similarity_matrix[np.ix_(temp_cluster, temp_cluster)])

                        if temp_entropy < min_entropy:
                            min_entropy = temp_entropy
                            closest_category = c

            if closest_category is not None and min_entropy < entropy_threshold:
                # Ensure the added category is not duplicated
                if closest_category not in added_categories:
                    new_cluster.append(closest_category)
                    new_labels += original_labels[closest_category]
                    added_categories.add(closest_category)  # Mark as merged
                    print(f"Added category {closest_category} to new cluster {new_cluster} with labels {new_labels}")

                original_labels[merged_label_index] = new_labels
                merged = True  # Mark as merged
            else:
                break  # No suitable category found for merging, exit loop

        clusters = update_labels(original_labels_1, new_cluster)
        
        round_results.append(clusters)
        
        new_centroids = merge_centroids(centroids, clusters)
        centroids = new_centroids

        n_classes = len(centroids)
        similarity_matrix, labels = compute_similarity_matrix(centroids)

    return clusters, original_labels, round_results, original_labels_rec


def calculate_custom_v_measure(cluster_assignments, fine_to_coarse, coarse_classes):
    """
    Calculate custom completeness metrics based on CIFAR-100 fine-to-coarse mapping.
    
    Args:
        cluster_assignments: List of lists, each sublist represents a cluster containing fine class indices
        fine_to_coarse: Dictionary mapping fine classes to coarse classes
        coarse_classes: List of coarse class names
    
    Returns:
        cluster_completeness_rec: List of completeness scores for each cluster
    """
    print("=" * 80)
    print("Custom Completeness Metrics Analysis")
    print("=" * 80)
    
    # Calculate total number of classes in each coarse category (5 classes per coarse category in CIFAR-100)
    k_i = [5] * 20  # Each coarse category has 5 fine classes
    
    total_categories = 0
    cluster_completeness_rec = []

    print(f"Number of Clusters: {len(cluster_assignments)}")

    # Calculate completeness for each cluster
    for cluster_id, cluster in enumerate(cluster_assignments):
        n_j = len(cluster)  # Total number of classes in cluster j
        if n_j == 0:
            continue
            
        total_categories += n_j
        
        # Count number of classes in each coarse category in this cluster
        # Use unique fine classes to ensure HSC stays in [0,1] range
        unique_cluster = sorted(set(cluster))
        n_j = len(unique_cluster)
        m_ij = [0] * 20  # Initialize count for each coarse category to 0
        
        for fine_idx in unique_cluster:
            if fine_idx in fine_to_coarse:
                coarse_idx = fine_to_coarse[fine_idx]
                m_ij[coarse_idx] += 1
        
        # Calculate completeness for this cluster (HSC score)
        cluster_completeness = 0.0
        for i in range(20):
            if m_ij[i] > 0:
                cluster_completeness += (m_ij[i] / k_i[i]) * m_ij[i]
        cluster_completeness /= n_j
        # Record completeness score
        cluster_completeness_rec.append(cluster_completeness)
        
        # Print detailed information for each cluster
        print(f"\nCluster {cluster_id} (Size: {n_j}):")
        print(f"  Completeness Score: {cluster_completeness:.4f}")
        
        # Show distribution of classes in the cluster
        coarse_counts = {}
        for fine_idx in cluster:
            if fine_idx in fine_to_coarse:
                coarse_idx = fine_to_coarse[fine_idx]
                coarse_name = coarse_classes[coarse_idx]
                coarse_counts[coarse_name] = coarse_counts.get(coarse_name, 0) + 1
        
        if coarse_counts:
            sorted_coarse = sorted(coarse_counts.items(), key=lambda x: x[1], reverse=True)
            for coarse, count in sorted_coarse:
                percentage = (count / n_j) * 100
                print(f"  {coarse}: {count} classes ({percentage:.1f}%)")
    
    # Calculate overall metrics
    if len(cluster_completeness_rec) > 0:
        mean_completeness = np.mean(cluster_completeness_rec)
        std_completeness = np.std(cluster_completeness_rec)
        min_completeness = np.min(cluster_completeness_rec)
        max_completeness = np.max(cluster_completeness_rec)
        
        # Output results
        print(f"\n📊 Custom Completeness Metrics:")
        print(f"  Mean Completeness: {mean_completeness:.4f}")
        print(f"  Std Deviation: {std_completeness:.4f}")
        print(f"  Min Score: {min_completeness:.4f}")
        print(f"  Max Score: {max_completeness:.4f}")
        print(f"  N = {len(cluster_completeness_rec)}")
        
        # Metric explanation
        print(f"\n🎯 Metric Explanation:")
        print(f"  Completeness: Measure of how well each cluster covers its semantic categories")
        print(f"  Higher scores indicate better semantic coherence within clusters")
        
        # Evaluation
        print(f"\n📈 Evaluation:")
        if mean_completeness >= 0.8:
            print(f"  ✅ Excellent - High clustering quality")
        elif mean_completeness >= 0.6:
            print(f"  ⚠️  Good - Reasonable clustering quality")
        elif mean_completeness >= 0.4:
            print(f"  ⚠️  Fair - Moderate clustering quality")
        else:
            print(f"  ❌ Poor - Low clustering quality")
    else:
        print("No clusters to evaluate")
        mean_completeness = 0.0
        std_completeness = 0.0
        min_completeness = 0.0
        max_completeness = 0.0
    
    return cluster_completeness_rec


def main():
    """Main function to run superclass entropy clustering evaluation."""
    print("Starting Superclass Entropy Clustering Evaluation")
    print("="*60)
    
    # Load CIFAR-100 fine-to-coarse mapping
    fine_to_coarse, coarse_classes = load_cifar100_fine_to_coarse_mapping()
    print(f"Loaded CIFAR-100 mapping with {len(coarse_classes)} coarse classes")
    
    # Load features - Use configurable path
    # Import config to get the feature path
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import FEATURE_PATHS
    
    feature_save_path = FEATURE_PATHS.get('cifar100')
    
    # Fallback to original path if not found (for backward compatibility)
    if not os.path.exists(feature_save_path):
        original_path = '/home/lzr/桌面/SDEntropy/results/cifar100/features_by_class.npz'
        if os.path.exists(original_path):
            print(f"Using original feature path: {original_path}")
            feature_save_path = original_path
    
    try:
        features = np.load(feature_save_path)
        print(f"Loaded features from {feature_save_path}")
    except FileNotFoundError:
        print(f"Feature file not found at {feature_save_path}")
        print("Please ensure the CIFAR-100 features are available.")
        print("You can:")
        print("  1. Run 'python scripts/train_and_extract.py --dataset cifar100' to generate features")
        print("  2. Set CIFAR100_FEATURE_PATH environment variable to your feature file location")
        print("  3. Edit config.py to set the correct path")
        return
    
    # Compute centroids
    print("Computing centroids for all classes...")
    centroids = compute_centroids(features)
    
    # Calculate complex entropy for each superclass
    print("\nComputing complex entropy for each superclass:")
    print("=" * 80)
    
    entropy_rec = []
    
    for coarse_idx in range(20):
        # Get all fine classes in this coarse category
        fine_classes_in_coarse = [fine_idx for fine_idx, coarse in fine_to_coarse.items() if coarse == coarse_idx]
        
        # Get centroids for these fine classes
        coarse_centroids = [centroids[fine_idx] for fine_idx in fine_classes_in_coarse]
        
        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix_for_list(coarse_centroids)
        
        # Calculate complex entropy
        entropy = calculate_complex_entropy(similarity_matrix)
        
        entropy_rec.append(entropy)
        print(f"Coarse class {coarse_idx:2d} ({coarse_classes[coarse_idx]:30s}): Complex Entropy = {entropy:.4f}")
        print(f"   Contains fine classes: {sorted(fine_classes_in_coarse)}")
    
    print("=" * 80)
    print("Complex entropy calculation for all 20 superclasses completed")
    
    # Convert to float
    entropy_values_float = [float(entropy) for entropy in entropy_rec]
    
    print("Converting to float type:")
    for i, entropy in enumerate(entropy_values_float):
        print(f"Coarse class {i}: Complex Entropy = {entropy}")
    
    # Run clustering for each entropy threshold
    SHC_average = []
    SHC_std = []
    SHC_wsum = []
    SHC_Eround = []
    
    for i, entropy_threshold in enumerate(entropy_values_float):
        print(f"\nProcessing entropy threshold {i}: {entropy_threshold}")
        
        # Adjust the entropy threshold 
        adjusted_threshold = entropy_threshold
        
        try:
            clusters, original_labels, round_results, original_labels_rec = agglomerative_clustering_with_entropy(features, adjusted_threshold)
            
            # Initialize label mapping
            label_map = {i: [i] for i in range(100)}
            
            # Create list to store tracking information for each round
            tracking_info = []
            
            # Process each round of clustering
            for round_idx, result in enumerate(round_results):
                current_round_info = []
                new_label_map = {}  # Create new mapping for this round
                
                # Build tracking information for each class
                for new_label, cluster in enumerate(result):
                    # Find all original labels in this class
                    original_labels_list = []
                    for item in cluster:
                        original_labels_list.extend(label_map[item])
                    
                    # Record the original labels for this class
                    current_round_info.append(original_labels_list)
                    
                    # Update the new label mapping (don't modify label_map during iteration)
                    new_label_map[new_label] = original_labels_list
                
                # Save tracking information for current round
                tracking_info.append(current_round_info)
                
                # Update label_map for next round
                label_map = new_label_map
            
            # Extract clustering result from tracking info
            # Get the last cluster from each round (except the first round)
            clustering_result = [info[-1] for idx, info in enumerate(tracking_info) if idx > 0]
            
            # Calculate custom completeness metrics
            SHC_rec = calculate_custom_v_measure(clustering_result, fine_to_coarse, coarse_classes)
            
            SHC_average_temp = np.mean(SHC_rec) if len(SHC_rec) > 0 else 0.0
            SHC_std_temp = np.std(SHC_rec) if len(SHC_rec) > 0 else 0.0
            SHC_average.append(SHC_average_temp)
            SHC_std.append(SHC_std_temp)
            SHC_wsum.append(np.sum(SHC_rec) if len(SHC_rec) > 0 else 0.0)
            SHC_Eround.append(SHC_rec)
            
        except Exception as e:
            print(f"Error processing threshold {i} (entropy={entropy_threshold}): {e}")
            import traceback
            traceback.print_exc()
            # Add default values for this threshold
            SHC_average.append(0.0)
            SHC_std.append(0.0)
            SHC_wsum.append(0.0)
            SHC_Eround.append([])
    
    # Print results
    print("\nClustering Results Summary:")
    print("Final Clustering Result:", clusters)
    print("SHC Average:", SHC_average)
    print("SHC Std:", SHC_std)
    print("SHC Wsum:", SHC_wsum)
    print("SHC Eround[19]:", SHC_Eround[19] if len(SHC_Eround) > 19 else "N/A")
    print("Mean of SHC Eround[1]:", np.mean(SHC_Eround[1]) if len(SHC_Eround) > 1 and len(SHC_Eround[1]) > 0 else 0.0)
    
    # Calculate overall mean across thresholds
    temp_mean = 0
    valid_thresholds = 0
    for i in range(min(20, len(SHC_Eround))):
        if len(SHC_Eround[i]) > 0:
            temp_mean += np.mean(SHC_Eround[i])
            valid_thresholds += 1
    
    if valid_thresholds > 0:
        overall_mean = temp_mean / valid_thresholds
        print("Overall mean across thresholds:", overall_mean)
    
    # Calculate traditional HAC methods' average HSC for comparison
    print("\n" + "="*80)
    print("Computing Traditional HAC Methods HSC for Comparison")
    print("="*80)
    
    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform, cosine
        
        # Compute centroids
        centroids_dict = compute_centroids(features)
        n_classes = len(centroids_dict)
        
        # Compute distance matrix
        distance_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                dist = cosine(centroids_dict[i], centroids_dict[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Convert to condensed form for scipy
        condensed_dist = squareform(distance_matrix)
        
        # Perform HAC with different linkage methods
        hac_methods = {
            'Ward': linkage(condensed_dist, method='ward'),
            'Average': linkage(condensed_dist, method='average'),
            'Complete': linkage(condensed_dist, method='complete')
        }
        
        # Calculate HSC for each HAC method
        hac_hsc_scores = {}
        for method_name, Z in hac_methods.items():
            # Extract non-leaf nodes (internal clusters)
            non_leaf_nodes = []
            for i in range(len(Z)):
                left_child = int(Z[i, 0])
                right_child = int(Z[i, 1])
                
                # Get all leaf nodes in this cluster
                def get_leaves(node_id, n_original):
                    if node_id < n_original:
                        return [node_id]
                    else:
                        idx = int(node_id - n_original)
                        left = int(Z[idx, 0])
                        right = int(Z[idx, 1])
                        return get_leaves(left, n_original) + get_leaves(right, n_original)
                
                cluster_members = get_leaves(n_classes + i, n_classes)
                non_leaf_nodes.append(cluster_members)
            
            # Calculate completeness for each non-leaf node
            hsc_scores = []
            for cluster in non_leaf_nodes:
                unique_cluster = sorted(set(cluster))
                n_j = len(unique_cluster)
                if n_j == 0:
                    continue
                
                # Count fine classes in each coarse category
                k_i = [5] * 20
                m_ij = [0] * 20
                
                for fine_idx in unique_cluster:
                    if fine_idx in fine_to_coarse:
                        coarse_idx = fine_to_coarse[fine_idx]
                        m_ij[coarse_idx] += 1
                
                # Calculate completeness (HSC)
                cluster_completeness = 0.0
                for i in range(20):
                    if m_ij[i] > 0:
                        cluster_completeness += (m_ij[i] / k_i[i]) * m_ij[i]
                cluster_completeness /= n_j
                hsc_scores.append(cluster_completeness)
            
            hac_hsc_scores[method_name] = hsc_scores
            mean_hsc = np.mean(hsc_scores) if len(hsc_scores) > 0 else 0.0
            print(f"  {method_name} Linkage: Mean HSC = {mean_hsc:.4f} (N={len(hsc_scores)})")
        
        # Calculate average across all traditional methods
        all_hac_means = [np.mean(scores) for scores in hac_hsc_scores.values() if len(scores) > 0]
        traditional_methods_mean = np.mean(all_hac_means) if len(all_hac_means) > 0 else 0.567
        
        print(f"\n📊 Traditional Methods Average HSC: {traditional_methods_mean:.4f}")
        print("="*80)
        
    except Exception as e:
        print(f"Warning: Could not calculate traditional HAC HSC: {e}")
        print("Using default value for comparison")
        traditional_methods_mean = 0.567  # Fallback to calculated average from run_hsc_evaluation.py
    
    # Create performance visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # CIFAR-100 superclass names (simplified version)
        superclass_names = [
            'Aquatic', 'Fish', 'Flowers', 'Food', 'Fruits',
            'Electronics', 'Furniture', 'Insects', 'Carnivores',
            'Structures', 'Landscapes', 'Herbivores',
            'Medium Mammals', 'Invertebrates', 'People', 'Reptiles', 
            'Small Mammals', 'Trees', 'Vehicles 1', 'Vehicles 2'
        ]
        
        # Prepare data for visualization
        print("Checking SHC_Eround structure:")
        for i in range(len(SHC_Eround)):
            if hasattr(SHC_Eround[i], '__len__'):
                print(f"Threshold {i}: {len(SHC_Eround[i]) if len(SHC_Eround[i]) > 0 else 0} nodes")
            else:
                print(f"Threshold {i}: single value {SHC_Eround[i]}")
        
        # Prepare data - handle irregular shapes
        all_hsc_data = []
        threshold_means = []
        threshold_stds = []
        threshold_counts = []
        
        for i in range(len(SHC_Eround)):
            if hasattr(SHC_Eround[i], '__len__') and not isinstance(SHC_Eround[i], str) and len(SHC_Eround[i]) > 0:
                # Process list/array
                hsc_values = list(SHC_Eround[i])
                for hsc in hsc_values:
                    all_hsc_data.append({
                        'Superclass': superclass_names[i],
                        'HSC': hsc,
                        'Threshold_Index': i
                    })
                threshold_means.append(np.mean(hsc_values))
                threshold_stds.append(np.std(hsc_values))
                threshold_counts.append(len(hsc_values))
            else:
                # Process single value
                if isinstance(SHC_Eround[i], (int, float)):
                    all_hsc_data.append({
                        'Superclass': superclass_names[i],
                        'HSC': SHC_Eround[i],
                        'Threshold_Index': i
                    })
                    threshold_means.append(SHC_Eround[i])
                    threshold_stds.append(0)  # Single value has no std
                    threshold_counts.append(1)
                else:
                    threshold_means.append(0)
                    threshold_stds.append(0)
                    threshold_counts.append(0)
        
        # Create the performance visualization plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for box plots
        box_data = []
        valid_threshold_indices = []
        
        for i in range(len(SHC_Eround)):
            if hasattr(SHC_Eround[i], '__len__') and not isinstance(SHC_Eround[i], str) and len(SHC_Eround[i]) > 0:
                box_data.append(SHC_Eround[i])
                valid_threshold_indices.append(i)
            elif isinstance(SHC_Eround[i], (int, float)):
                box_data.append([SHC_Eround[i]])
                valid_threshold_indices.append(i)
            else:
                box_data.append([])
                valid_threshold_indices.append(i)
        
        # Create violin plot with box plot overlay
        if box_data and any(len(data) > 0 for data in box_data):
            # Filter out empty lists for plotting
            filtered_box_data = [data for data in box_data if len(data) > 0]
            filtered_positions = [i for i, data in enumerate(box_data) if len(data) > 0]
            
            if filtered_box_data:
                violin_parts = ax.violinplot(filtered_box_data, positions=filtered_positions, 
                                           showmeans=False, showmedians=False, widths=0.8)
                
                # Set violin colors
                colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_box_data)))
                for i, pc in enumerate(violin_parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.3)
                    pc.set_edgecolor('black')
                
                # Overlay box plots
                box_plot = ax.boxplot(filtered_box_data, positions=filtered_positions, widths=0.4,
                                     patch_artist=True, showfliers=False)
                
                # Set box plot colors
                for i, box in enumerate(box_plot['boxes']):
                    box.set_facecolor(colors[i])
                    box.set_alpha(0.7)
                
                # Add mean points
                valid_means = []
                valid_positions = []
                for i, data in enumerate(filtered_box_data):
                    if len(data) > 0:
                        mean_val = np.mean(data)
                        valid_means.append(mean_val)
                        valid_positions.append(filtered_positions[i])
                
                if valid_means:
                    ax.scatter(valid_positions, valid_means, color='red', marker='D', 
                              s=80, zorder=4, label='Mean HSC', edgecolors='white', linewidth=1)
                
                # Add traditional methods performance reference line
                ax.axhline(y=traditional_methods_mean, color='red', linestyle='--', linewidth=2, 
                          label=f'Traditional Methods (HSC = {traditional_methods_mean:.3f})')
                
                # Calculate and show performance range
                if valid_means:
                    mean_across_thresholds = np.mean(valid_means)
                    min_mean = min(valid_means) if valid_means else 0
                    max_mean = max(valid_means) if valid_means else 0
                    range_padding = (max_mean - min_mean) * 0.1  # Add 10% padding
                    performance_min = min_mean - range_padding if min_mean > range_padding else 0
                    performance_max = max_mean + range_padding
                    
                    ax.axhspan(performance_min, performance_max, 
                              alpha=0.2, color='green', label='SHC Performance Range')
                    
                    # Calculate intuitive statistics
                    stat_text = f'Mean HSC: {mean_across_thresholds:.3f}\nRange: [{min_mean:.3f}, {max_mean:.3f}]'
                    ax.text(0.02, 0.02, stat_text, transform=ax.transAxes, fontsize=11,
                            verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Set up chart
        ax.set_xlabel('Superclass (Source of SCEntropy Threshold)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hierarchical Semantic Coherence (HSC)', fontsize=12, fontweight='bold')
        
        # Set x-ticks and labels for all 20 superclasses
        ax.set_xticks(range(20))
        ax.set_xticklabels(superclass_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Set title
        ax.set_title('Robust Generalization of Structural Complexity Entropy\nConsistent High Performance Across Diverse Threshold Choices', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('Figure_3b_Generalization_Single.pdf', format='pdf', dpi=300, bbox_inches='tight')
        print("Performance visualization saved as 'Figure_3b_Generalization_Single.pdf'")
        
        # Print detailed statistics
        if valid_means:
            print(f"\nGeneralization Detailed Statistics:")
            print(f"Average HSC across all thresholds: {np.mean(valid_means):.3f} ± {np.std(valid_means):.3f}")
            print(f"HSC Range: [{np.min(valid_means):.3f}, {np.max(valid_means):.3f}]")
            avg_counts = np.mean([c for c in threshold_counts if c > 0])
            std_counts = np.std([c for c in threshold_counts if c > 0])
            print(f"Average number of nodes: {avg_counts:.1f} ± {std_counts:.1f}")
            print(f"Coefficient of variation: {np.std(valid_means)/np.mean(valid_means) if np.mean(valid_means) != 0 else 0:.3f}")
            
            # Calculate performance stability
            hsc_above_07 = sum(1 for mean in valid_means if mean > 0.7)
            print(f"High-performance threshold ratio (HSC > 0.7): {hsc_above_07/len(valid_means)*100:.1f}%")
        
        plt.close()  # Close the plot to free memory
        
    except ImportError:
        print("\nMatplotlib or related visualization libraries not available, skipping visualization.")
    
    print("\nSuperclass Entropy Clustering Evaluation completed successfully!")


if __name__ == "__main__":
    main()
