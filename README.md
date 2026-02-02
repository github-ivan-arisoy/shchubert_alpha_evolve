# SchuBERT Experiment Pipeline

A self-contained experimental setup for the HuBERT foundation model, designed for AlphaEvolve-like agent sweeps and rapid experimentation.

## Overview

This pipeline combines the complete HuBERT training in a single runnable file:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SchuBERT Training Pipeline                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐         │
│  │ Step 1: Pretrain │   │ Step 2: Finetune │   │ Step 3: Downstream│         │
│  │   (Foundation)   │──▶│  (Specialization)│──▶│   (Final Head)   │         │
│  │                  │   │                  │   │                  │         │
│  │   • MLM Loss     │   │   • Binary CTR   │   │ • Frozen Backbone│         │
│  │   • NT-Xent Loss │   │   • R², Log-loss │   │ • New 2x Dense(32)│         │
│  └──────────────────┘   └──────────────────┘   │ • Test w/ Bias   │         │
│                                                └──────────────────┘         │
│                                                         │                    │
│                                                         ▼                    │
│                                              ┌──────────────────────┐       │
│                                              │   CLS Embeddings     │       │
│                                              │   for Production     │       │
│                                              └──────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Stages

1. **Pre-training (Foundation)**: Learn user history representations
   - **MLM Loss**: Masked Language Modeling - predict masked tokens
   - **NT-Xent Loss**: Contrastive learning between two views of the same user
   - **No validation**: Training only

2. **Fine-tuning (Specialization)**: Specialize for click intent prediction
   - **Binary Classification**: Predict click probability
   - Uses pretrained backbone with trainable weights
   - **Metrics**: AUC, Log-loss, R²
   - **Validation**: Required (ideally final hour of dataset, fallback: 1% random split)

3. **Downstream (Final Head)**: Train lightweight CTR head
   - **Frozen Backbone**: All pretrained/finetuned layers are frozen 🥶
   - **New Trainable Head**: Two Dense(32) layers + sigmoid
   - **Metrics**: AUC, Log-loss, R², RIG
   - **Test evaluation** with bias shift for negative sampling compensation

## Directory Structure

```
hubert_experiment/
├── hubert_experiment.py    # Main training script (all-in-one)
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data/
│   ├── pretrain/
│   │   └── train/          # Pre-training parquet files (no validation)
│   │       └── *.parquet
│   ├── finetune/
│   │   ├── train/          # Fine-tuning training data
│   │   │   └── *.parquet
│   │   └── val/            # Validation (ideally: final hour of dataset)
│   │       └── *.parquet   # If empty: 1% random split from train
│   └── downstream/
│       ├── train/          # Downstream training data
│       │   └── *.parquet
│       ├── val/            # Validation (ideally: final hour of dataset)
│       │   └── *.parquet   # If empty: 1% random split from train
│       └── test/           # Test data for final evaluation
│           └── *.parquet
├── models/
│   ├── pretrain/           # Saved pre-trained models
│   │   └── hubert_ntxent_model/
│   ├── finetune/           # Saved fine-tuned models
│   │   └── hubert_ctr_finetuned/
│   └── downstream/         # Saved downstream models
│       └── hubert_downstream_ctr/
└── logs/
    ├── pretrain/           # Pre-training metrics CSV
    │   └── pretrain_metrics.csv
    ├── finetune/           # Fine-tuning metrics CSV
    │   └── finetune_metrics.csv
    └── downstream/         # Downstream metrics CSV
        ├── downstream_metrics.csv
        └── test_metrics.csv  # Final test evaluation metrics
```

## Validation Strategy

### Fine-tuning and Downstream

The validation set is **required** for both fine-tuning and downstream training:

| Priority | Source | Description |
|----------|--------|-------------|
| 1st | `val/` folder | Parquet files in the validation subfolder |
| 2nd | 1% random split | Automatic fallback if `val/` folder is empty |

**Best Practice**: Use the **final hour** of your training dataset as validation. This simulates real production conditions where the model is trained on historical data and evaluated on the most recent data.

```bash
# Example: Split your data by time
# - Training: All data except last hour
# - Validation: Last hour of data
```

If no validation data is provided, the system will automatically perform a random 1% split from training data (with a warning).

## Metrics and Evaluation

### Fine-tuning Metrics

| Metric | Description | Logged |
|--------|-------------|--------|
| loss | Binary cross-entropy loss | ✅ train + val |
| auc | Area Under ROC Curve | ✅ train + val |
| log_loss | Log-loss (same as loss) | ✅ train + val |
| r2 | R² coefficient of determination | ✅ train + val |

### Downstream Metrics

**Training/Validation:**

| Metric | Description | Logged |
|--------|-------------|--------|
| loss | Binary cross-entropy loss | ✅ train + val |
| auc | Area Under ROC Curve | ✅ train + val |
| log_loss | Log-loss | ✅ train + val |
| r2 | R² score | ✅ train + val |

**Test Evaluation:**

| Metric | Description | Logged |
|--------|-------------|--------|
| test_auc | AUC on test set | ✅ test_metrics.csv |
| test_log_loss | Log-loss on test set | ✅ test_metrics.csv |
| test_r2 | R² on test set | ✅ test_metrics.csv |
| test_rig | Relative Information Gain | ✅ test_metrics.csv |

### Test Evaluation with Bias Shift

The downstream test evaluation includes a **bias shift** to compensate for negative sampling:

```python
# The test data should use 10% negative sampling (configurable)
# The model's final layer bias is adjusted:
new_bias = old_bias + log(negative_sampling_rate / 100)
```

This ensures the model outputs calibrated probabilities even when trained on undersampled data.

**RIG (Relative Information Gain)** is computed as:
```
RIG = (baseline_loss - model_loss) / baseline_loss
```
Where `baseline_loss` is the loss when predicting the mean of the training labels.

## Quick Start

### 1. Generate Dummy Data (for testing)

```bash
cd hubert_experiment
python hubert_experiment.py --mode generate-dummy
```

This creates sample parquet files with the new folder structure:
- `data/pretrain/train/*.parquet`
- `data/finetune/train/*.parquet` + `data/finetune/val/*.parquet`
- `data/downstream/train/*.parquet` + `data/downstream/val/*.parquet` + `data/downstream/test/*.parquet`

### 2. Run Full Pipeline

```bash
# Run all 3 steps: pre-training → fine-tuning → downstream
python hubert_experiment.py --mode all
```

### 3. Run Individual Stages

```bash
# Pre-training only (MLM + NT-Xent)
python hubert_experiment.py --mode pretrain

# Fine-tuning only (requires pretrained model)
python hubert_experiment.py --mode finetune

# Downstream only (requires fine-tuned model)
python hubert_experiment.py --mode downstream
```

## Data Format

### Pre-training Data (`data/pretrain/train/*.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `sentence_a` | `list[int]` | Token IDs for view A (masked) |
| `sentence_b` | `list[int]` | Token IDs for view B (masked) |
| `sentence_a_times` | `list[float]` | Time delays for view A |
| `sentence_b_times` | `list[float]` | Time delays for view B |
| `labels_a` | `list[int]` | MLM labels for A (-1 = not masked) |
| `labels_b` | `list[int]` | MLM labels for B (-1 = not masked) |
| `sequence_length` | `int` | Sequence length (same for all rows) |
| `vocabulary_size` | `int` | Vocabulary size (same for all rows) |

### Fine-tuning & Downstream Data

| Column | Type | Description |
|--------|------|-------------|
| `sentence_a` | `list[int]` | Token IDs for user history |
| `sentence_a_times` | `list[float]` | Time delays for each token |
| `target_click` | `int` | Binary click label (0 or 1) |
| `sequence_length` | `int` | Sequence length (same for all rows) |
| `vocabulary_size` | `int` | Vocabulary size (same for all rows) |

**Note**: Column names `input_a` and `input_a_times` are also supported for backwards compatibility.

## CSV Logging

All training metrics are saved to CSV files for analysis:

### Pre-training: `logs/pretrain/pretrain_metrics.csv`
```csv
epoch,loss,mlm_loss,contrastive_loss
1,5.2341,3.1234,2.1107
2,4.8765,2.8901,1.9864
...
```

### Fine-tuning: `logs/finetune/finetune_metrics.csv`
```csv
epoch,loss,auc,log_loss,r2,val_loss,val_auc,val_log_loss,val_r2
1,0.2341,0.6234,0.2341,0.0123,0.2456,0.6012,0.2456,0.0089
2,0.1987,0.6789,0.1987,0.0234,0.2123,0.6345,0.2123,0.0156
...
```

### Downstream: `logs/downstream/downstream_metrics.csv`
```csv
epoch,loss,auc,log_loss,r2,val_loss,val_auc,val_log_loss,val_r2
1,0.1234,0.7123,0.1234,0.0345,0.1456,0.6890,0.1456,0.0289
...
```

### Test Metrics: `logs/downstream/test_metrics.csv`
```csv
test_auc,test_log_loss,test_r2,test_rig
0.7234,0.1234,0.0456,0.0234
```

## Command Line Options

```bash
python hubert_experiment.py --help
```

### Mode Selection
| Argument | Values | Description |
|----------|--------|-------------|
| `--mode` | `all`, `pretrain`, `finetune`, `downstream`, `generate-dummy` | Training mode |

### Model Architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--embed-dim` | 16 | Embedding dimension |
| `--ff-dim` | 64 | Feed-forward dimension |
| `--nb-head` | 2 | Number of attention heads |
| `--nb-transformer-blocs` | 2 | Number of transformer blocks |

### Training Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 1024 | Batch size for pretrain/finetune |
| `--epochs` | 10 | Number of epochs (all stages) |
| `--learning-rate` | 0.001 | Learning rate for pre-training |
| `--finetune-learning-rate` | 0.001 | Learning rate for fine-tuning |

### Downstream-Specific Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--downstream-batch-size` | 15000 | Batch size for downstream |
| `--downstream-learning-rate` | 0.001 | Learning rate for downstream |
| `--downstream-hidden-dim` | 32 | Hidden dim for new Dense layers |

### Data Paths
| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrain-data-dir` | `data/pretrain` | Pre-training data directory |
| `--finetune-data-dir` | `data/finetune` | Fine-tuning data directory |
| `--downstream-data-dir` | `data/downstream` | Downstream data directory |

### Model Paths
| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrain-model-dir` | `models/pretrain` | Pre-trained model output directory |
| `--finetune-model-dir` | `models/finetune` | Fine-tuned model output directory |
| `--downstream-model-dir` | `models/downstream` | Downstream model output directory |

## Example Commands

### Quick Test Run
```bash
# Generate dummy data and run with minimal epochs
python hubert_experiment.py --mode generate-dummy
python hubert_experiment.py --mode all --epochs 2
```

### Production-like Run
```bash
# Full training with custom parameters
python hubert_experiment.py --mode all \
    --embed-dim 128 \
    --ff-dim 512 \
    --nb-head 4 \
    --nb-transformer-blocs 4 \
    --epochs 20 \
    --batch-size 512 \
    --downstream-batch-size 15000
```

### Using Your Own Data
```bash
# Place your parquet files in the data directories:
# - data/pretrain/train/*.parquet
# - data/finetune/train/*.parquet, data/finetune/val/*.parquet
# - data/downstream/train/*.parquet, data/downstream/val/*.parquet, data/downstream/test/*.parquet

python hubert_experiment.py --mode all
```

## Configuration

All parameters are centralized at the top of `hubert_experiment.py` for easy modification:

```python
# 🏗️  MODEL ARCHITECTURE (shared across all steps)
MODEL_CONFIG = {
    "embed_dim": 16,
    "ff_dim": 64,
    "nb_head": 2,
    "nb_transformer_blocs": 2,
}

# 📚  STEP 1: PRE-TRAINING
PRETRAIN_CONFIG = {
    "batch_size": 1024,
    "epochs": 1,
    "learning_rate": 0.001,
    ...
}

# 🎯  STEP 2: FINE-TUNING
FINETUNE_CONFIG = {
    "batch_size": 1024,
    "epochs": 1,
    "learning_rate": 0.001,
    "val_split_fallback": 0.01,  # 1% if val folder is empty
    ...
}

# 🧊  STEP 3: DOWNSTREAM
DOWNSTREAM_CONFIG = {
    "batch_size": 15000,
    "epochs": 1,
    "learning_rate": 0.001,
    "hidden_dim": 32,
    "val_split_fallback": 0.01,
    "negative_sampling_rate": 10,  # 10% for test bias shift
    ...
}
```

## Downstream Architecture Details

The downstream step creates a model with:

```
┌─────────────────────────────────────────────┐
│           Frozen Backbone (from finetune)    │
│  • word_embedding (frozen)                   │
│  • time_embedding (frozen)                   │
│  • position_embedding (frozen)               │
│  • shared_encoder (frozen)                   │
│  • cls_ctr (frozen) → CLS embedding          │
└──────────────────────┬──────────────────────┘
                       │ CLS embedding (embed_dim)
                       ▼
┌─────────────────────────────────────────────┐
│           New Trainable Head                 │
│  • Dense(32, relu) - downstream_dense_1     │
│  • Dense(32, relu) - downstream_dense_2     │
│  • Dense(1, sigmoid) - downstream_click_pred│
└─────────────────────────────────────────────┘
```

**Key**: Only the 3 new Dense layers are trained. All backbone parameters are frozen.

## Model Saving

Models are saved as **TensorFlow SavedModel folders** (not zipped):
- `models/pretrain/hubert_ntxent_model/`
- `models/finetune/hubert_ctr_finetuned/`
- `models/downstream/hubert_downstream_ctr/`

This avoids I/O errors with zip operations on some filesystems.

## AlphaEvolve Integration

This setup is designed for automated agent sweeps. Key modification points:

### Loss Function (lines ~380-420)
```python
# Modify NT-Xent temperature, loss weighting, etc.
def nt_xent_loss(self, z_i, z_j, temperature: float = 0.1):
    ...
```

### Model Architecture (lines ~460-550)
```python
# Modify transformer blocks, embedding dimensions, etc.
def create_bert_model(self):
    ...
```

### Downstream Head (lines ~1180-1220)
```python
# Modify the new trainable layers
x = layers.Dense(self.params.hidden_dim, activation="relu", ...)(cls_embedding)
x = layers.Dense(self.params.hidden_dim, activation="relu", ...)(x)
click_pred = layers.Dense(1, activation="sigmoid", ...)(x)
```

## Porting Changes Back

After agent sweeps, modifications can be ported back to the original codebase:

1. **Pre-training changes** → `lib-python-ml-hubert-v5/users/users/models/bert/hubert_model.py`
2. **Fine-tuning changes** → `lib-python-ml/users/users/models/bert/hubert_model.py`
3. **Downstream changes** → New file to be created based on patterns

The code structure mirrors the original files for easy diff comparison.

## Requirements

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
pyarrow>=8.0.0
scikit-learn>=1.0.0
```

## Troubleshooting

### Memory Issues
- Reduce `--batch-size` (e.g., 64 or 128)
- Reduce `--downstream-batch-size` if needed
- Reduce `--embed-dim` for testing

### GPU Issues
- The pipeline uses `tf.distribute.MirroredStrategy` for multi-GPU
- For single GPU, this works transparently

### Missing Data
- Run `--mode generate-dummy` first to create sample data
- Check that parquet files are in the correct directories (with train/val/test subfolders)

### Missing Validation Data
- The system will use a 1% random split from training data
- For best results, provide actual validation data (final hour of dataset)
- A warning will be logged if using the fallback

### I/O Errors
- Models are now saved as folders, not zip files
- This avoids `OSError: [Errno 5] Input/output error` on some filesystems

## File Correspondence to Original Codebase

| This File Section | Original Location |
|-------------------|-------------------|
| `BertTrainer` | `lib-python-ml-hubert-v5/.../hubert_model.py:BertTrainer` |
| `HuBertPretrainer` | `lib-python-ml-hubert-v5/.../hubert_model.py:HuBertModel` |
| `HuBertFinetuner` | `lib-python-ml/.../hubert_model.py:HuBertModel` |
| `HuBertDownstream` | New (not in original codebase) |
| `LinearTimeEmbedding` | Both files (shared layer) |
| `R2Score` | New custom Keras metric |
| `compute_classification_metrics` | New utility function |
