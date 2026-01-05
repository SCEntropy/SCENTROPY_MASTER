# Quick Start Guide for Reviewers

This guide helps paper reviewers quickly set up and run the experiments.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster training)
- ~10GB disk space for datasets and results

## Quick Setup (5 minutes)

```bash
# 1. Clone and enter the project
git clone <repository-url>
cd SDEntropy_master

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# That's it! No additional configuration needed.
```

## Running Experiments

### Visual Concept Discovery (Main Results)

```bash
cd SCEnrtopy_in_VisualConceptDiscovery

# Run complete pipeline for each dataset (recommended)
python scripts/run_experiment.py --dataset fashionmnist --epochs 10
python scripts/run_experiment.py --dataset cifar10 --epochs 10
python scripts/run_experiment.py --dataset cifar100 --epochs 10

# Or run individual components
python scripts/train_and_extract.py --dataset cifar100 --epochs 10
python scripts/run_hsc_evaluation.py
python scripts/run_superclass_entropy.py
```

**Expected Runtime:**
- FashionMNIST: ~10-15 minutes
- CIFAR-10: ~15-20 minutes
- CIFAR-100: ~30-40 minutes

### NLG Clustering (Supplementary)

```bash
cd SCEntropy_in_NLG/NL_clustering

# Test sentence clustering with sample data
python NL_clustering.py --sentences-file QUESTION1.txt
```

## Verifying Results

### Visual Module - Expected Outputs

1. **Feature files**: `results/{dataset}/features_by_class.npz`
2. **Clustering results**: Console output showing hierarchical structure
3. **Evaluation figures**: 
   - `Figure_3a_HSC_Distribution.pdf` (HSC comparison)
   - `Figure_3b_Generalization_Single.pdf` (Robustness evaluation)

### Key Metrics to Check

- **FashionMNIST**: Should discover semantic groups (upper-body clothing, footwear, bags)
- **CIFAR-10**: Should separate animals from vehicles with high accuracy
- **CIFAR-100**: HSC scores should show advantage over traditional HAC methods

## Troubleshooting

### Issue: Feature file not found

**Solution**: Train and extract features first:
```bash
python scripts/train_and_extract.py --dataset cifar100 --epochs 10
```

### Issue: Out of memory

**Solution**: Reduce batch size:
```bash
python scripts/run_experiment.py --dataset cifar100 --batch_size 32
```

### Issue: CUDA not available

**Solution**: Code automatically falls back to CPU (will be slower)

## Path Configuration (Optional)

The project uses **relative paths by default** - no configuration needed!

If you want to use custom paths:

**Option 1: Environment Variable**
```bash
export CIFAR100_FEATURE_PATH=/path/to/features_by_class.npz
python scripts/run_hsc_evaluation.py
```

**Option 2: Configuration File**
Edit `SCEnrtopy_in_VisualConceptDiscovery/config.py`:
```python
FEATURE_PATHS = {
    'cifar100': '/your/custom/path/features_by_class.npz'
}
```

## Reproducibility Notes

- Results may show slight variations due to training randomness
- For exact reproduction, use the same random seed and pre-extracted features
- The clustering algorithm itself is deterministic given the same features

## Getting Help

If you encounter issues:
1. Check the main README.md for detailed documentation
2. Ensure all dependencies are correctly installed
3. Verify you have sufficient disk space and memory
4. Check that Python 3.8+ is being used

## Expected Paper Results

After running all experiments, you should be able to verify:

1. **Table 1**: Clustering results for three datasets
2. **Figure 3a**: HSC distribution showing SHC advantage
3. **Figure 3b**: Generalization across entropy thresholds
4. **Qualitative results**: Semantic hierarchy visualization

Total time to reproduce all results: **~1-2 hours** (depending on hardware)
