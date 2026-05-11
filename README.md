# PCVRHyFormer

A hybrid Transformer model for **Post-Click Conversion Rate (PCVR)** prediction. Combines Non-Sequence (NS) feature tokenization with multi-domain sequence encoding via stacked `MultiSeqHyFormerBlock` layers.

## Model Architecture

```
Input Features
├── User Int Features ──→ NS Tokenizer ──→ user NS tokens
├── User Dense Features ────────────────→ user dense token
├── Item Int Features ──→ NS Tokenizer ──→ item NS tokens
├── Item Dense Features ────────────────→ item dense token
└── Sequences (seq_a/b/c/d)
    ├── Sequence Embedding + Time Bucket
    └── Seq Encoder (Transformer / SwiGLU / Longer)
        └── Cross-Attention with Query tokens
            ↓
    MultiSeqHyFormerBlock × N
    (NS tokens + Query tokens → RankMixer → Self-Attention → FFN)
            ↓
    Classifier → P(conversion)
```

**Key components:**

- **NS Tokenizer**: Two variants — `group` (project each feature group to one token) or `rankmixer` (concatenate all embeddings, split into equal-size chunks)
- **Sequence Encoder**: Three variants — `transformer` (standard self-attention), `swiglu` (SwiGLU without attention), `longer` (Top-K compressed encoder)
- **RoPE**: Optional Rotary Position Embedding for sequence attention
- **RankMixer**: Token mixing block with per-token FFN
- **Dual Optimizer**: Adagrad for sparse Embeddings, AdamW for dense parameters
- **Early Stopping**: Monitors validation AUC

## Project Structure

```
├── model.py          # Model definition (PCVRHyFormer, RoPE, SwiGLU, etc.)
├── dataset.py        # Parquet dataset loader with IterableDataset
├── trainer.py        # Training loop with dual optimizer & early stopping
├── train.py          # Training entry point with CLI argument parsing
├── utils.py          # Utilities (seed, EarlyStopping, logging, focal loss)
├── ns_groups.json    # Example NS feature grouping config
├── run.sh            # Shell script to launch training
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch >= 1.12
- PyArrow
- scikit-learn
- tqdm
- TensorBoard

```bash
pip install torch pyarrow scikit-learn tqdm tensorboard
```

## Data Format

Training data should be placed in a directory containing:

1. **`*.parquet` files** — One or more Parquet files with the following columns:

   | Column | Type | Description |
   |---|---|---|
   | `user_int_feats_*` | int / int[] | User categorical features (e.g. user_id, age_bucket) |
   | `user_dense_feats_*` | float / float[] | User numerical features (e.g. statistics) |
   | `item_int_feats_*` | int / int[] | Item categorical features (e.g. item_id, category) |
   | `item_dense_feats_*` | float / float[] | Item numerical features |
   | `seq_a`, `seq_b`, `seq_c`, `seq_d` | int[][] | User behavior sequences (e.g. click, fav, cart, purchase) |
   | `seq_a_len`, `seq_b_len`, ... | int | Actual length of each sequence |
   | `seq_a_time_bucket`, `seq_b_time_bucket`, ... | int[] | Time delta buckets for each sequence position |
   | `label` | int | Binary conversion label (0 or 1) |

2. **`schema.json`** — Feature metadata describing each column's `feature_id`, `offset`, and `length` in the flattened tensor.

Example directory layout:
```
data/
├── train_001.parquet
├── train_002.parquet
├── ...
└── schema.json
```

## Quick Start

### Option 1: Using `run.sh`

```bash
bash run.sh --data_dir /path/to/your/data
```

### Option 2: Manual launch

```bash
python train.py \
    --data_dir /path/to/your/data \
    --batch_size 256 \
    --lr 1e-4 \
    --num_epochs 10 \
    --device cuda
```

### Option 3: Using environment variables

```bash
export TRAIN_DATA_PATH=/path/to/your/data
export TRAIN_CKPT_PATH=./checkpoints
export TRAIN_LOG_PATH=./logs
python train.py
```

## CLI Arguments

### Paths

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | None | Training data directory (env: `TRAIN_DATA_PATH`) |
| `--schema_path` | `<data_dir>/schema.json` | Schema JSON path |
| `--ckpt_dir` | `./checkpoints` | Checkpoint output directory (env: `TRAIN_CKPT_PATH`) |
| `--log_dir` | `./logs` | Log directory (env: `TRAIN_LOG_PATH`) |

### Training Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 256 | Batch size |
| `--lr` | 1e-4 | Learning rate for dense params (AdamW) |
| `--sparse_lr` | 0.05 | Learning rate for sparse params (Adagrad) |
| `--num_epochs` | 999 | Max epochs (early stopping typically terminates earlier) |
| `--patience` | 5 | Early stopping patience |
| `--seed` | 42 | Random seed |
| `--device` | cuda/cpu | Training device |

### Model Architecture

| Argument | Default | Description |
|---|---|---|
| `--d_model` | 64 | Backbone hidden dimension |
| `--emb_dim` | 64 | Per-embedding-table dimension |
| `--num_queries` | 1 | Query tokens per sequence domain |
| `--num_hyformer_blocks` | 2 | Number of stacked HyFormer blocks |
| `--num_heads` | 4 | Attention heads |
| `--seq_encoder_type` | transformer | Sequence encoder: `transformer`, `swiglu`, `longer` |
| `--hidden_mult` | 4 | FFN inner-dim multiplier |
| `--dropout_rate` | 0.01 | Dropout rate |
| `--rank_mixer_mode` | full | RankMixer: `full`, `ffn_only`, `none` |
| `--use_rope` | false | Enable RoPE positional encoding |

### NS Tokenizer

| Argument | Default | Description |
|---|---|---|
| `--ns_tokenizer_type` | rankmixer | `group` or `rankmixer` |
| `--ns_groups_json` | `./ns_groups.json` | NS feature grouping config |
| `--user_ns_tokens` | 0 | User NS tokens in rankmixer mode (0 = auto) |
| `--item_ns_tokens` | 0 | Item NS tokens in rankmixer mode (0 = auto) |

### Loss Function

| Argument | Default | Description |
|---|---|---|
| `--loss_type` | bce | `bce` (BCEWithLogits) or `focal` (Focal Loss) |
| `--focal_alpha` | 0.1 | Focal Loss alpha (only when loss_type=focal) |
| `--focal_gamma` | 2.0 | Focal Loss gamma (only when loss_type=focal) |

### Data Pipeline

| Argument | Default | Description |
|---|---|---|
| `--num_workers` | 16 | DataLoader workers |
| `--buffer_batches` | 20 | Shuffle buffer size (in batches) |
| `--train_ratio` | 1.0 | Fraction of training Row Groups to use |
| `--valid_ratio` | 0.1 | Fraction of Row Groups for validation |
| `--seq_max_lens` | seq_a:256,seq_b:256,seq_c:512,seq_d:512 | Per-domain sequence truncation |

## NS Groups Configuration

`ns_groups.json` defines how integer features are grouped for the NS Tokenizer. The file has two top-level keys:

```json
{
  "user_ns_groups": {
    "U1": [1, 15],
    "U2": [48, 49, 89, 90, 91]
  },
  "item_ns_groups": {
    "I1": [11, 13],
    "I2": [5, 6, 7, 8, 12]
  }
}
```

Values are `feature_id`s (the numeric suffix in column names like `user_int_feats_48`). `train.py` converts them to schema entry indices at runtime. If the file is missing, each feature is placed in its own singleton group.

**Token count formula**: `T = num_queries * num_sequences + num_ns`

For the example config: `T = 1 * 4 + (7 + 1 + 4) = 16`. `d_model` must be divisible by `T` (e.g. `d_model=64` works).

## Checkpoints

Each checkpoint directory contains:

- `model.pt` — Model weights
- `schema.json` — Copy of the feature schema (for inference)
- `ns_groups.json` — Copy of the NS groups config (if provided)
- `train_config.json` — Full training configuration

## License

Private / Internal Use.
