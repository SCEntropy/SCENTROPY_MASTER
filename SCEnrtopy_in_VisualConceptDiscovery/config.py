"""
Configuration file for SCEntropy in Visual Concept Discovery project.
Contains default values and configuration parameters.
"""
import os

# Path Configuration
# For reproducibility: You can either use the default relative paths or set absolute paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Default configuration parameters
DEFAULT_CONFIG = {
    'dataset': 'fashionmnist',
    'epochs': 10,
    'batch_size': 64,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'entropy_threshold': 0.4,  # For fashionmnist and cifar10
    'entropy_threshold_cifar100': 6.0,  # Specific threshold for cifar100
    'data_dir': './data/',
    'results_dir': './results/',
    'device': None,  # Will be set to 'cuda' if available, otherwise 'cpu'
    'num_workers': 2,
    'model_name': 'resnet50',
    'pretrained': True
}

# Dataset-specific configurations
DATASET_CONFIGS = {
    'fashionmnist': {
        'num_classes': 10,
        'entropy_threshold': 0.4
    },
    'cifar10': {
        'num_classes': 10,
        'entropy_threshold': 0.4
    },
    'cifar100': {
        'num_classes': 100,
        'entropy_threshold': 6.0
    }
}

# Feature file paths for reproducibility evaluation
# Option 1: Use relative path (recommended for reviewers/users)
# Option 2: Set environment variable FEATURE_PATH to override
# Option 3: Uncomment and set absolute path below
FEATURE_PATHS = {
    'cifar100': os.environ.get('CIFAR100_FEATURE_PATH', 
                               os.path.join(PROJECT_ROOT, 'results/cifar100/features_by_class.npz'))
}

# For backward compatibility with original experiments
# If you have the original feature files at a specific location, set this
# ORIGINAL_FEATURE_PATH = '/path/to/original/SDEntropy/results/cifar100/features_by_class.npz'