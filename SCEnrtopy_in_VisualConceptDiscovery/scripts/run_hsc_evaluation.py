#!/usr/bin/env python3
"""
Hierarchical Semantic Coherence (HSC) Evaluation Script

This script implements the evaluation of clustering results using custom completeness metrics,
comparing different hierarchical clustering methods (Ward, Average, Complete) with the 
Structural Complexity Entropy method.
"""
import numpy as np
import os
import sys
from typing import Dict, List, Tuple

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


def load_non_leaf_nodes(data_dir: str = "results/cifar100"):
    """
    Load non-leaf nodes from clustering results.
    
    Args:
        data_dir: Directory containing clustering results
        
    Returns:
        Dictionary with non-leaf nodes for each method
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform, cosine
    import numpy as np
    import sys
    import os
    
    # Load actual feature data from the original notebook
    # Import config to get the feature path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config import FEATURE_PATHS
    
    feature_save_path = FEATURE_PATHS.get('cifar100')
    
    # Fallback to original path if not found (for backward compatibility)
    if not os.path.exists(feature_save_path):
        original_path = "/home/lzr/桌面/SDEntropy/results/cifar100/features_by_class.npz"
        if os.path.exists(original_path):
            print(f"Using original feature path: {original_path}")
            feature_save_path = original_path
    
    try:
        # Load features from the file
        features = np.load(feature_save_path)
        print(f"Loaded features from {feature_save_path}")
    except FileNotFoundError:
        print(f"Feature file not found at {feature_save_path}")
        print("Using sample data instead...")
        # Generate sample feature centroids for 100 CIFAR-100 classes
        np.random.seed(42)  # For reproducible results
        n_classes = 100
        n_features = 512  # Typical feature dimension for ResNet50
        
        # Generate random centroids that simulate actual feature distributions
        # Each class has a centroid in feature space
        centroids = {}
        for i in range(n_classes):
            # Generate slightly different distributions per class to simulate real features
            centroids[i] = np.random.normal(loc=i % 10, scale=1.0, size=n_features)
        
        # Create a sample distance matrix based on these centroids
        labels = sorted(centroids.keys())
        n = len(labels)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Use Euclidean distance between centroids
                    distance_matrix[i, j] = np.linalg.norm(centroids[labels[i]] - centroids[labels[j]])
                else:
                    distance_matrix[i, j] = 0.0
    else:
        # Calculate centroids from actual features
        def compute_centroids(features):
            centroids = {}
            for class_label in features:
                centroids[int(class_label)] = np.mean(features[class_label], axis=0)
            return centroids
        
        # Calculate distance matrix using cosine distance (as in original notebook)
        def compute_distance_matrix(centroids):
            labels = sorted(centroids.keys())
            n = len(labels)
            distance_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        distance_matrix[i, j] = cosine(centroids[labels[i]], centroids[labels[j]])
                    else:
                        distance_matrix[i, j] = 0.0
            return distance_matrix, labels
        
        centroids = compute_centroids(features)
        print(f"Computed centroids for {len(centroids)} classes")
        
        # Calculate distance matrix
        distance_matrix, labels = compute_distance_matrix(centroids)
        print("Distance matrix computed using cosine distance")
    
    # Function to extract non-leaf nodes from linkage matrix
    def extract_non_leaf_nodes(Z, labels):
        """
        Extract all non-leaf nodes from hierarchical clustering linkage matrix.
        
        Args:
            Z: linkage matrix
            labels: original labels list
            
        Returns:
            List of non-leaf nodes, each containing the leaf nodes it encompasses
        """
        n = len(labels)
        non_leaf_nodes = []
        
        # Create a mapping to store what each node (including leaf and non-leaf) contains
        node_contents = {}
        
        # Initialize leaf nodes
        for i in range(n):
            node_contents[i] = [labels[i]]
        
        # Process each non-leaf node (merging step)
        for i in range(len(Z)):
            # Current non-leaf node index
            node_id = n + i
            
            # Get the two child nodes being merged
            left_child = int(Z[i, 0])
            right_child = int(Z[i, 1])
            
            # Get the classes contained in each child node
            left_classes = node_contents[left_child]
            right_classes = node_contents[right_child]
            
            # Combine to get all classes in current node
            current_classes = sorted(left_classes + right_classes)
            
            # Store current node's content
            node_contents[node_id] = current_classes
            
            # Add current non-leaf node to results
            non_leaf_nodes.append(current_classes)
        
        return non_leaf_nodes
    
    # Perform hierarchical clustering with different linkage methods
    methods = ['ward', 'average', 'complete']
    clustering_results = {}
    
    # Convert distance matrix to condensed form for scipy
    condensed_dist = squareform(distance_matrix, checks=False)
    
    for method in methods:
        try:
            # Perform hierarchical clustering
            Z = linkage(condensed_dist, method=method)
            
            # Extract non-leaf nodes
            non_leaf_nodes = extract_non_leaf_nodes(Z, labels)
            
            # Filter to only include nodes with multiple classes
            filtered_nodes = [node for node in non_leaf_nodes if len(node) > 1]
            
            clustering_results[method] = filtered_nodes
        except Exception as e:
            print(f"Error in {method} clustering: {e}")
            # Fallback to empty list if clustering fails
            clustering_results[method] = []
    
    # Our method clustering result (from original notebook)
    our_method_result = [
        [55, 72, 3, 93],
        [47, 52, 96, 59, 33],
        [11, 35, 2, 46, 98],
        [50, 74, 64, 80, 63, 36],
        [30, 95, 73],
        [62, 92, 70, 54, 51],
        [5, 25, 84, 20, 94],
        [4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75],
        [10, 61, 28],
        [26, 45],
        [78, 99],
        [13, 81, 90, 58],
        [32, 67, 30, 95, 73, 91, 1],
        [27, 44],
        [4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75, 27, 44, 32, 67, 30, 95, 73, 91, 1, 26, 45, 38, 65],
        [7, 24, 6, 79],
        [42, 88, 43],
        [4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75, 27, 44, 32, 67, 30, 95, 73, 91, 1, 26, 45, 38, 65, 42, 88, 43, 97, 34, 11, 35, 2, 46, 98],
        [15, 19, 31, 4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75, 27, 44, 32, 67, 30, 95, 73, 91, 1, 26, 45, 38, 65, 42, 88, 43, 97, 34, 11, 35, 2, 46, 98, 21],
        [40, 5, 25, 84, 20, 94, 87],
        [18, 7, 24, 6, 79, 77, 78, 99],
        [14, 18, 7, 24, 6, 79, 77, 78, 99, 62, 92, 70, 54, 51, 15, 19, 31, 4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75, 27, 44, 32, 67, 30, 95, 73, 91, 1, 26, 45, 38, 65, 42, 88, 43, 97, 34, 11, 35, 2, 46, 98, 21, 82],
        [23, 71, 60, 49],
        [86, 40, 5, 25, 84, 20, 94, 87, 22, 14, 18, 7, 24, 6, 79, 77, 78, 99, 62, 92, 70, 54, 51, 15, 19, 31, 4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75, 27, 44, 32, 67, 30, 95, 73, 91, 1, 26, 45, 38, 65, 42, 88, 43, 97, 34, 11, 35, 2, 46, 98, 21, 82, 39],
        [10, 61, 28, 86, 40, 5, 25, 84, 20, 94, 87, 22, 14, 18, 7, 24, 6, 79, 77, 78, 99, 62, 92, 70, 54, 51, 15, 19, 31, 4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75, 27, 44, 32, 67, 30, 95, 73, 91, 1, 26, 45, 38, 65, 42, 88, 43, 97, 34, 11, 35, 2, 46, 98, 21, 82, 39],
        [16, 10, 61, 28, 86, 40, 5, 25, 84, 20, 94, 87, 22, 14, 18, 7, 24, 6, 79, 77, 78, 99, 62, 92, 70, 54, 51, 15, 19, 31, 4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75, 27, 44, 32, 67, 30, 95, 73, 91, 1, 26, 45, 38, 65, 42, 88, 43, 97, 34, 11, 35, 2, 46, 98, 21, 82, 39, 9],
        [56, 47, 52, 96, 59, 33, 23, 71, 60, 49],
        [17, 37, 56, 47, 52, 96, 59, 33, 23, 71, 60, 49],
        [0, 57, 83, 53, 16, 10, 61, 28, 86, 40, 5, 25, 84, 20, 94, 87, 22, 14, 18, 7, 24, 6, 79, 77, 78, 99, 62, 92, 70, 54, 51, 15, 19, 31, 4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75, 27, 44, 32, 67, 30, 95, 73, 91, 1, 26, 45, 38, 65, 42, 88, 43, 97, 34, 11, 35, 2, 46, 98, 21, 82, 39, 9],
        [12, 17, 37, 56, 47, 52, 96, 59, 33, 23, 71, 60, 49, 76, 69, 68],
        [13, 81, 90, 58, 12, 17, 37, 56, 47, 52, 96, 59, 33, 23, 71, 60, 49, 76, 69, 68, 85, 0, 57, 83, 53, 16, 10, 61, 28, 86, 40, 5, 25, 84, 20, 94, 87, 22, 14, 18, 7, 24, 6, 79, 77, 78, 99, 62, 92, 70, 54, 51, 15, 19, 31, 4, 55, 72, 3, 93, 50, 74, 64, 80, 63, 36, 66, 29, 75, 27, 44, 32, 67, 30, 95, 73, 91, 1, 26, 45, 38, 65, 42, 88, 43, 97, 34, 11, 35, 2, 46, 98, 21, 82, 39, 9]
    ]
    
    return {
        'ward': clustering_results['ward'],
        'average': clustering_results['average'],
        'complete': clustering_results['complete'],
        'our_method': our_method_result
    }


def main():
    """
    Main function to run HSC evaluation.
    """
    print("Starting Hierarchical Semantic Coherence (HSC) Evaluation")
    print("="*60)
    
    # Load CIFAR-100 fine-to-coarse mapping
    fine_to_coarse, coarse_classes = load_cifar100_fine_to_coarse_mapping()
    print(f"Loaded CIFAR-100 mapping with {len(coarse_classes)} coarse classes")
    
    # Initialize HSC Metrics calculator
    hsc_metrics = HSCMetrics(fine_to_coarse, coarse_classes)
    
    # Load clustering results (non-leaf nodes)
    clustering_results = load_non_leaf_nodes()
    
    print(f"Loaded clustering results for {len(clustering_results)} methods")
    
    # Prepare method results for comparison
    method_results = {
        'HAC Ward Linkage': clustering_results['ward'],
        'HAC Average Linkage': clustering_results['average'],
        'HAC Complete Linkage': clustering_results['complete'],
        'SHC (Ours)': clustering_results['our_method']
    }
    
    # Compare methods using completeness metrics
    all_stats = hsc_metrics.compare_methods(method_results)
    
    # Print final comparison summary
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'N':<6} {'Mean':<8} {'Std':<8} {'<0.5%':<8}")
    print("-" * 70)
    
    for method_name, stats in all_stats.items():
        print(f"{method_name:<25} {stats['n']:<6} {stats['mean']:<8.3f} {stats['std']:<8.3f} {stats['perc_below_05']:<8.1f}%")
    
    # Identify best method
    best_method = max(all_stats.items(), key=lambda x: x[1]['mean'])
    print(f"\n🏆 Best Method: {best_method[0]} (Mean Score: {best_method[1]['mean']:.3f})")
    
    # Generate visualization plots
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        # Prepare data for visualization
        ward_rec = all_stats['HAC Ward Linkage']['completeness_scores']
        average_rec = all_stats['HAC Average Linkage']['completeness_scores']
        complete_rec = all_stats['HAC Complete Linkage']['completeness_scores']
        SHC_rec = all_stats['SHC (Ours)']['completeness_scores']
        
        # Calculate statistics for visualization
        def calculate_stats(data, method_name):
            n = len(data)
            mean_val = np.mean(data)
            std_val = np.std(data)
            perc_below_05 = (np.array(data) < 0.5).sum() / n * 100
            return {
                'method': method_name,
                'n': n,
                'mean': mean_val,
                'std': std_val,
                'perc_below_05': perc_below_05
            }
        
        methods_stats = [
            calculate_stats(ward_rec, 'Ward'),
            calculate_stats(average_rec, 'Average'),
            calculate_stats(complete_rec, 'Complete'),
            calculate_stats(SHC_rec, 'SHC (Ours)')
        ]
        
        # Create violin plot visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create violin plot
        violin_parts = ax.violinplot([ward_rec, average_rec, complete_rec, SHC_rec], 
                                    positions=[0, 1, 2, 3], showmeans=False, showmedians=True)
        
        # Set violin plot colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        # Set median line style
        violin_parts['cmedians'].set_color('black')
        violin_parts['cmedians'].set_linewidth(2)
        
        # Overlay box plot (showing medians and quartiles only)
        box_parts = ax.boxplot([ward_rec, average_rec, complete_rec, SHC_rec], 
                              positions=[0, 1, 2, 3], widths=0.15, patch_artist=True,
                              showfliers=False, medianprops={'color': 'black', 'linewidth': 2})
        
        # Set box plot colors
        for i, box in enumerate(box_parts['boxes']):
            box.set_facecolor(colors[i])
            box.set_alpha(0.9)
            box.set_edgecolor('black')
        
        # Overlay scatter plot to show data point distribution
        df = pd.DataFrame({
            'Method': ['HAC Ward'] * len(ward_rec) + ['HAC Average'] * len(average_rec) + 
                      ['HAC Complete'] * len(complete_rec) + ['SHC (Ours)'] * len(SHC_rec),
            'HSC Score': list(ward_rec) + list(average_rec) + list(complete_rec) + list(SHC_rec)
        })
        
        for i, (method, color) in enumerate(zip(['HAC Ward', 'HAC Average', 'HAC Complete', 'SHC (Ours)'], colors)):
            data = df[df['Method'] == method]['HSC Score']
            # Add random jitter to avoid overlapping
            x_jitter = np.random.normal(i, 0.05, len(data))
            ax.scatter(x_jitter, data, alpha=0.6, color=color, s=30, edgecolors='white', linewidth=0.5)
        
        # Add mean and sample count annotations using calculated stats
        for i, stats in enumerate(methods_stats):
            # Use fixed-width text box
            text = f'{stats["method"]}\nHSC = {stats["mean"]:.3f}\nn = {stats["n"]}'
            
            # Use boxstyle to set fixed size
            ax.text(3.8, 0.85 - i*0.15, text, 
                    fontsize=10, va='center', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
        
        # Set axis and labels
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Ward\nLinkage', 'Average\nLinkage', 'Complete\nLinkage', 'SHC\n(Ours)'], 
                          fontsize=12, fontweight='bold')
        ax.set_ylabel('Hierarchical Semantic Coherence (HSC)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Clustering Method', fontsize=14, fontweight='bold')
        
        # Set y-axis range - ensure it doesn't exceed 1.0
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        # Set x-axis range to leave space for text boxes on the right
        ax.set_xlim(-0.5, 4.5)
        
        # Add grid lines
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add horizontal reference line
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Set title
        ax.set_title('Distribution of Hierarchical Semantic Coherence (HSC) Scores\non CIFAR-100 Dataset', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Create detailed legend with all method-specific information and red dashed line explanation
        from matplotlib.patches import Patch
        import matplotlib.lines as mlines
        
        legend_elements = [
            Patch(facecolor=colors[0], alpha=0.7, label=f'Ward: {methods_stats[0]["perc_below_05"]:.1f}% < 0.5'),
            Patch(facecolor=colors[1], alpha=0.7, label=f'Average: {methods_stats[1]["perc_below_05"]:.1f}% < 0.5'),
            Patch(facecolor=colors[2], alpha=0.7, label=f'Complete: {methods_stats[2]["perc_below_05"]:.1f}% < 0.5'),
            Patch(facecolor=colors[3], alpha=0.7, label=f'SHC (Ours): {methods_stats[3]["perc_below_05"]:.1f}% < 0.5'),
            mlines.Line2D([], [], color='red', linestyle='--', label='HSC = 0.5 threshold')
        ]
        
        # Place legend at bottom right
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save as PDF format
        plt.savefig('Figure_3a_HSC_Distribution.pdf', format='pdf', dpi=300, bbox_inches='tight')
        
        # Skip showing the plot as per requirements
        print("Plot visualization is disabled as per requirements")
        print("Plot saved as 'Figure_3a_HSC_Distribution.pdf'")
        
        # Close the plot to free memory
        plt.close()
        
        # Print calculated statistics
        print("\nCalculated Statistics:")
        for stats in methods_stats:
            print(f"{stats['method']}: n={stats['n']}, Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, <0.5 percentage={stats['perc_below_05']:.1f}%")
            
    except ImportError:
        print("\nMatplotlib or related visualization libraries not available, skipping visualization.")
    
    print("\nHSC Evaluation completed successfully!")


if __name__ == "__main__":
    main()