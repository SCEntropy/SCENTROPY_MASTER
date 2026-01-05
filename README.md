# SCEntropy: Structural Complexity Entropy Applications

This repository contains implementations of Structural Complexity Entropy (SCEntropy) methods applied to two different domains: **Visual Concept Discovery** in image classification and **Natural Language Processing** for text clustering and evaluation.

## 📖 For Paper Reviewers

**New to this project?** Start with our [Quick Start Guide for Reviewers (QUICKSTART.md)](QUICKSTART.md) for a streamlined setup and experiment walkthrough.

## Overview

The project demonstrates the versatility of entropy-based clustering algorithms across different data modalities:

1. **SCEntropy in Visual Concept Discovery**: Automatic discovery of semantic hierarchies in visual concepts from image datasets (FashionMNIST, CIFAR-10, CIFAR-100)
2. **SCEntropy in NLG**: Semantic sentence clustering and text quality evaluation using entropy-based algorithms

## Project Structure

```
SDEntropy_master/
├── README.md                                    # This file
├── requirements.txt                             # Combined dependencies for all modules
│
├── SCEntropy_in_VisualConceptDiscovery/        # Visual concept discovery module
│   ├── src/                                     # Source code
│   │   ├── models/                              # Model definitions and training
│   │   ├── clustering/                          # Clustering algorithms
│   │   ├── data_processing/                     # Data loading and preprocessing
│   │   ├── utils/                               # Utility functions
│   │   └── visualization/                       # Plotting and visualization
│   ├── scripts/                                 # Experiment scripts
│   │   ├── run_experiment.py                    # Main experiment runner
│   │   ├── train_and_extract.py                 # Model training and feature extraction
│   │   ├── print_clustering_hierarchy.py        # Display clustering results
│   │   ├── run_hsc_evaluation.py                # HSC metrics evaluation
│   │   └── run_superclass_entropy.py            # Superclass entropy robustness
│   ├── results/                                 # Output directory
│   └── data/                                    # Dataset directory
│
└── SCEntropy_in_NLG/                            # Natural language processing module
    ├── NL_clustering/                           # Sentence clustering tool
    │   ├── NL_clustering.py                     # Main entry point
    │   ├── src/                                 # Source code
    │   │   ├── main_processor.py                # Processing workflow
    │   │   ├── text_processor.py                # Text cleaning and logging
    │   │   ├── embedding_clustering.py          # Embeddings and clustering
    │   │   └── ai_interface.py                  # AI model interfaces
    │   ├── all-MiniLM-L6-v2/                    # Local sentence transformer model
    │   └── QUESTION*.txt                        # Sample input files
    │
    └── Performance/                             # Text evaluation suite
        ├── input/                               # Shared evaluation data
        ├── bert/                                # BERTScore evaluator
        ├── meteor/                              # METEOR evaluator
        └── rouge/                               # ROUGE evaluator
```

## Installation

### Prerequisites

- Python 3.8+
- Git
- Internet connection for model downloads (optional for offline setup)

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd SDEntropy_master
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   This will install all required packages for both visual and NLG modules.

4. (Optional) Configure paths for reproducibility:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env to set custom paths if needed
   # This is optional - the project works with default relative paths
   ```

## Module 1: SCEntropy in Visual Concept Discovery

### Overview

Implements entropy-based clustering algorithms to **discover hierarchical semantic structures** in visual concepts from image datasets. The core innovation is using structural complexity entropy to guide automatic discovery of conceptual relationships without predefined taxonomies.

### Key Features

- **Automatic Visual Concept Discovery**: Novel entropy-based approach without predefined taxonomies
- **Non-binary Tree Clustering**: Multi-cluster simultaneous merging for natural concept organization
- **Multi-dataset Support**: FashionMNIST, CIFAR-10, and CIFAR-100 with consistent English output
- **Hierarchical Semantic Coherence (HSC) Evaluation**: Custom metrics for evaluating discovered hierarchies
- **Superclass Entropy Robustness Testing**: Performance evaluation across different entropy thresholds
- Feature extraction using pre-trained ResNet-50 models
- Similarity matrix computation (cosine similarity and Euclidean distance)
- Visualization tools for discovered hierarchies

### Usage

#### Running the Complete Pipeline

```bash
cd SCEntropy_in_VisualConceptDiscovery

# Run experiment for a specific dataset
python scripts/run_experiment.py --dataset fashionmnist --epochs 10 --entropy_threshold 0.4
python scripts/run_experiment.py --dataset cifar10 --epochs 10 --entropy_threshold 0.4
python scripts/run_experiment.py --dataset cifar100 --epochs 10 --entropy_threshold 6.0
```

#### Running Specific Components

```bash
# Train model and extract features
python scripts/train_and_extract.py --dataset cifar10 --epochs 10

# Display clustering hierarchy (non-binary tree structure)
python scripts/print_clustering_hierarchy.py

# Run Hierarchical Semantic Coherence (HSC) evaluation
python scripts/run_hsc_evaluation.py

# Run Superclass Entropy Robustness Evaluation
python scripts/run_superclass_entropy.py
```

#### Configuration Options

- `--dataset`: Dataset to use (fashionmnist, cifar10, cifar100)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 64)
- `--entropy_threshold`: Threshold for entropy-based clustering (default: 0.4)
- `--data_dir`: Directory containing datasets (default: ./data/)
- `--results_dir`: Directory for saving results (default: ./results/)

#### Path Configuration for Reproducibility

For reviewers and users who want to reproduce the paper results, the project supports flexible path configuration:

**Option 1: Use Default Relative Paths (Recommended)**

By default, the project uses relative paths within the project directory:
```bash
# Features will be saved/loaded from:
SCEntropy_in_VisualConceptDiscovery/results/cifar100/features_by_class.npz
```

No additional configuration needed. Just run:
```bash
# First, train and extract features
python scripts/train_and_extract.py --dataset cifar100 --epochs 10

# Then run evaluation scripts
python scripts/run_hsc_evaluation.py
python scripts/run_superclass_entropy.py
```

**Option 2: Use Environment Variables**

Set the `CIFAR100_FEATURE_PATH` environment variable to specify a custom feature file location:
```bash
export CIFAR100_FEATURE_PATH=/path/to/your/features_by_class.npz
python scripts/run_hsc_evaluation.py
```

**Option 3: Edit Configuration File**

Modify `SCEntropy_in_VisualConceptDiscovery/config.py`:
```python
# Uncomment and set the path in config.py
ORIGINAL_FEATURE_PATH = '/path/to/original/features_by_class.npz'
```

**For Paper Reviewers:**

If you want to verify results against the original paper experiments:
1. Use Option 1 (recommended): Train from scratch using the provided code
2. Or use Option 2/3 if you have access to the original pre-extracted features

The evaluation scripts (`run_hsc_evaluation.py` and `run_superclass_entropy.py`) will automatically:
- Try to load features from the configured path
- Fall back to generating sample data if features are not found
- Provide clear instructions on how to generate the required features

#### Results

The pipeline generates:
1. Trained models and extracted deep features
2. Similarity matrices (cosine and Euclidean) revealing conceptual relationships
3. Discovered semantic hierarchies with different granularity levels
4. Hierarchical Semantic Coherence (HSC) evaluation metrics

Results are saved in `SCEntropy_in_VisualConceptDiscovery/results/` under dataset-specific subdirectories.

#### Reproducing Paper Figures

**Figure 3a - HSC Distribution Comparison:**
```bash
python scripts/run_hsc_evaluation.py
```
Generates `Figure_3a_HSC_Distribution.pdf` comparing HSC scores between traditional HAC methods and the proposed SHC method.

**Figure 3b - Generalization Performance:**
```bash
python scripts/run_superclass_entropy.py
```
Generates `Figure_3b_Generalization_Single.pdf` demonstrating robustness across 20 different entropy thresholds.

### Reproducibility Notes

Due to randomness in neural network training, **slight variations in clustering results may occur** across different runs. This is expected behavior.

**Common sources of randomness:**
- Random weight initialization
- Data augmentation (e.g., RandomHorizontalFlip)
- Batch shuffling during training
- Non-deterministic CUDA operations

**Expected behavior by dataset:**
- **CIFAR-10**: Highly reproducible - results typically match 100% due to high inter-class distinguishability
- **FashionMNIST**: Minor variations in early clustering rounds, but semantic groupings remain correct
- **CIFAR-100**: Variations in clustering sequence, but final semantic categories are consistent

**For exact reproducibility**, use pre-extracted features from original experiments. The clustering algorithm itself is deterministic given the same input features.

## Module 2: SCEntropy in NLG

### Overview

A comprehensive platform for sentence clustering and text quality evaluation using entropy-based algorithms. Consists of two main components: **NL_clustering** for semantic sentence clustering and **Performance Evaluation Suite** for multi-metric quality assessment.

### Key Features

#### NL_clustering (Sentence Clustering)
- Semantic sentence clustering using entropy-based algorithms
- **QUESTION format support**: Parse files with multiple sections
- Semantic embeddings using all-MiniLM-L6-v2 sentence transformer
- Entropy-based agglomerative clustering
- Text-based dendrogram output for visualization
- Configurable entropy threshold
- Auto-detection of file format

#### Performance Evaluation Suite
- Content extraction from source files
- AI-powered sentence rewriting using DeepSeek API
- Multi-metric evaluation: BERTScore, METEOR, and ROUGE
- **Batch evaluation mode**: Process multiple files automatically
- Centralized data storage in `Performance/input/` folder
- Unified data format across all evaluators

### Usage

#### Sentence Clustering

```bash
cd SCEntropy_in_NLG/NL_clustering

# Basic usage with auto-format detection
python NL_clustering.py --sentences-file QUESTION1.txt

# With custom entropy threshold
python NL_clustering.py --sentences-file QUESTION1.txt --entropy-threshold 0.8

# Specify output file and embedding model
python NL_clustering.py --sentences-file QUESTION1.txt --output-file results.txt --embedding-model all-MiniLM-L6-v2
```

**Input File Formats:**

*Format 1: Label format (simple sentences)*
```
Label 1: First sentence content here.
Label 2: Second sentence content here.
```

*Format 2: QUESTION format (multiple sections)*
```
###Question###
Your question text here

###Raw data###
Label 0: Raw sentence 1
Label 1: Raw sentence 2

###Constrained generation results###
Label 0: Constrained result 1

###Unconstrained generation result###
Label 0: Unconstrained result 1
```

**Configuration Options:**
- `--sentences-file`: Path to .txt file containing sentences (required)
- `--entropy-threshold`: Entropy threshold for clustering (default: 1.0)
- `--output-file`: Output file for logging (default: output.txt)
- `--embedding-model`: Sentence transformer model path (default: all-MiniLM-L6-v2)
- `--question-mode`: Enable QUESTION file parsing mode

#### Performance Evaluation

```bash
cd SCEntropy_in_NLG/Performance

# Step 1: Run BERTScore evaluation
cd bert
python bert_evaluator.py                    # Batch mode: process all files
python bert_evaluator.py --input-file ../input/data1.txt  # Single file mode

# Step 2: Run METEOR evaluation
cd ../meteor
python meteor_evaluator.py                  # Batch mode
python meteor_evaluator.py --input-file ../input/data1.txt  # Single file

# Step 3: Run ROUGE evaluation
cd ../rouge
python rouge_evaluator.py                   # Batch mode
python rouge_evaluator.py --input-file ../input/data1.txt  # Single file
```

**Available Metrics:**
- `bert`: BERTScore - Semantic similarity using BERT embeddings (Precision, Recall, F1)
- `meteor`: METEOR - Translation quality with stemming and synonymy
- `rouge`: ROUGE - N-gram overlap and sequence matching (ROUGE-1, ROUGE-2, ROUGE-L)

### API Configuration (Optional)

Only the Performance Extract module requires API keys for sentence rewriting:

**Environment Variables:**
- `DEEPSEEK_API_KEY`: DeepSeek API key
- `DEEPSEEK_BASE_URL`: API base URL (default: https://api.deepseek.com)
- `DEEPSEEK_MODEL`: Model name (default: deepseek-chat)

Create a `.env` file in the Extract directory with these variables, or pass them as command-line arguments.

**Note:** The sentence clustering (NL_clustering) module works entirely offline using local models.

### Model Configuration

**NL_clustering Models:**
- **all-MiniLM-L6-v2**: Sentence transformer for semantic embeddings (can be used locally)

**Performance Models:**
- **BERT**: For BERTScore calculations (configurable via --model-type or --local-model-path)

Models can be downloaded automatically from Hugging Face or placed locally in the project directory.

#### Manual Model Download

If you are in an offline environment:

**For all-MiniLM-L6-v2:**
1. Visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
2. Download all files (config.json, pytorch_model.bin, tokenizer.json, etc.)
3. Place in `SCEntropy_in_NLG/NL_clustering/all-MiniLM-L6-v2/`

**For BERT model:**
1. Visit: https://huggingface.co/bert-base-uncased
2. Download all files
3. Place in `SCEntropy_in_NLG/Performance/bert/bert-base-uncased/`

### Results

#### NL_clustering Output
- Console output with detailed processing logs
- Output file (output.txt) with complete results
- For QUESTION format: `{filename}_dendrograms.txt` with clustering visualizations

#### Performance Evaluation Output
- **BERTScore**: Precision, Recall, F1 scores in text files
- **METEOR**: Average and individual scores for both constrained and unconstrained
- **ROUGE**: Per-sample results with ROUGE-1, ROUGE-2, ROUGE-L metrics

## Reproducibility

### Visual Module
Due to training randomness, slight variations may occur. Use pre-extracted features for exact reproducibility. The clustering algorithm is deterministic given same input features.

### NLG Module
Due to AI-generated content, results may vary across runs. For consistent results, use pre-generated data files. Evaluation metrics are deterministic given same input data.

## Contributing

This project is research-focused. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Citation

If you use this code in your research, please cite the original paper.

## License

This project follows standard open-source practices. Please see individual module directories for specific license information.
