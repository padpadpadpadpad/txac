# PCVRHyFormer

一个用于**点击后转化率预测（Post-Click Conversion Rate, PCVR）**的混合 Transformer 模型。通过堆叠的 `MultiSeqHyFormerBlock` 层，将非序列（NS）特征 tokenize 与多域序列编码相结合。

## 模型架构

```
输入特征
├── 用户整数特征 ──→ NS Tokenizer ──→ 用户 NS tokens
├── 用户稠密特征 ────────────────→ 用户稠密 token
├── 物品整数特征 ──→ NS Tokenizer ──→ 物品 NS tokens
├── 物品稠密特征 ────────────────→ 物品稠密 token
└── 行为序列 (seq_a/b/c/d)
    ├── 序列 Embedding + 时间桶
    └── 序列编码器 (Transformer / SwiGLU / Longer)
        └── 与 Query tokens 做交叉注意力
            ↓
    MultiSeqHyFormerBlock × N
    (NS tokens + Query tokens → RankMixer → Self-Attention → FFN)
            ↓
    分类器 → P(转化)
```

**核心组件：**

- **NS Tokenizer**：两种变体 —— `group`（将每个特征组投影为一个 token）或 `rankmixer`（拼接所有 embedding 后等分切块）
- **序列编码器**：三种变体 —— `transformer`（标准自注意力）、`swiglu`（无注意力的 SwiGLU）、`longer`（Top-K 压缩编码器）
- **RoPE**：可选的旋转位置编码，用于序列注意力
- **RankMixer**：Token 混合模块，包含逐 token 的 FFN
- **双重优化器**：稀疏 Embedding 使用 Adagrad，稠密参数使用 AdamW
- **早停机制**：监控验证集 AUC

## 项目结构

```
├── model.py          # 模型定义（PCVRHyFormer、RoPE、SwiGLU 等）
├── dataset.py        # Parquet 数据集加载器（IterableDataset）
├── trainer.py        # 训练循环（双重优化器 + 早停）
├── train.py          # 训练入口（CLI 参数解析）
├── utils.py          # 工具函数（随机种子、EarlyStopping、日志、Focal Loss）
├── ns_groups.json    # NS 特征分组配置示例
├── run.sh            # 训练启动脚本
└── README.md
```

## 环境依赖

- Python 3.8+
- PyTorch >= 1.12
- PyArrow
- scikit-learn
- tqdm
- TensorBoard

```bash
pip install torch pyarrow scikit-learn tqdm tensorboard
```

## 数据格式

训练数据需要放在一个目录下，包含以下内容：

1. **`*.parquet` 文件** —— 一个或多个 Parquet 文件，包含以下列：

   | 列名 | 类型 | 说明 |
   |---|---|---|
   | `user_int_feats_*` | int / int[] | 用户类别特征（如 user_id、年龄段等） |
   | `user_dense_feats_*` | float / float[] | 用户数值特征（如统计值） |
   | `item_int_feats_*` | int / int[] | 物品类别特征（如 item_id、类目等） |
   | `item_dense_feats_*` | float / float[] | 物品数值特征 |
   | `seq_a`, `seq_b`, `seq_c`, `seq_d` | int[][] | 用户行为序列（如点击、收藏、加购、购买） |
   | `seq_a_len`, `seq_b_len`, ... | int | 各序列的实际长度 |
   | `seq_a_time_bucket`, `seq_b_time_bucket`, ... | int[] | 各序列位置的时间间隔桶 |
   | `label` | int | 转化标签（0 或 1） |

2. **`schema.json`** —— 特征元数据，描述每列的 `feature_id`、`offset` 和 `length`（在展平张量中的位置）。

目录结构示例：
```
data/
├── train_001.parquet
├── train_002.parquet
├── ...
└── schema.json
```

## 快速开始

### 方式一：使用 `run.sh`

```bash
bash run.sh --data_dir /path/to/your/data
```

### 方式二：手动启动

```bash
python train.py \
    --data_dir /path/to/your/data \
    --batch_size 256 \
    --lr 1e-4 \
    --num_epochs 10 \
    --device cuda
```

### 方式三：使用环境变量

```bash
export TRAIN_DATA_PATH=/path/to/your/data
export TRAIN_CKPT_PATH=./checkpoints
export TRAIN_LOG_PATH=./logs
python train.py
```

## 命令行参数

### 路径配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | 无 | 训练数据目录（环境变量：`TRAIN_DATA_PATH`） |
| `--schema_path` | `<data_dir>/schema.json` | Schema JSON 路径 |
| `--ckpt_dir` | `./checkpoints` | 模型检查点输出目录（环境变量：`TRAIN_CKPT_PATH`） |
| `--log_dir` | `./logs` | 日志目录（环境变量：`TRAIN_LOG_PATH`） |

### 训练超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--batch_size` | 256 | 批大小 |
| `--lr` | 1e-4 | 稠密参数学习率（AdamW） |
| `--sparse_lr` | 0.05 | 稀疏参数学习率（Adagrad） |
| `--num_epochs` | 999 | 最大训练轮数（通常由早停提前终止） |
| `--patience` | 5 | 早停耐心值（验证集无改善的轮数） |
| `--seed` | 42 | 随机种子 |
| `--device` | cuda/cpu | 训练设备 |

### 模型架构

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--d_model` | 64 | 骨干网络隐藏维度 |
| `--emb_dim` | 64 | 每个 Embedding 表的维度 |
| `--num_queries` | 1 | 每个序列域的 Query token 数量 |
| `--num_hyformer_blocks` | 2 | 堆叠的 HyFormer Block 层数 |
| `--num_heads` | 4 | 注意力头数 |
| `--seq_encoder_type` | transformer | 序列编码器：`transformer`、`swiglu`、`longer` |
| `--hidden_mult` | 4 | FFN 内部维度倍数 |
| `--dropout_rate` | 0.01 | Dropout 比率 |
| `--rank_mixer_mode` | full | RankMixer 模式：`full`、`ffn_only`、`none` |
| `--use_rope` | false | 启用 RoPE 位置编码 |

### NS Tokenizer

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ns_tokenizer_type` | rankmixer | `group` 或 `rankmixer` |
| `--ns_groups_json` | `./ns_groups.json` | NS 特征分组配置文件 |
| `--user_ns_tokens` | 0 | rankmixer 模式下用户 NS token 数（0 = 自动） |
| `--item_ns_tokens` | 0 | rankmixer 模式下物品 NS token 数（0 = 自动） |

### 损失函数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--loss_type` | bce | `bce`（BCEWithLogits）或 `focal`（Focal Loss） |
| `--focal_alpha` | 0.1 | Focal Loss 的 alpha（仅 loss_type=focal 时生效） |
| `--focal_gamma` | 2.0 | Focal Loss 的 gamma（仅 loss_type=focal 时生效） |

### 数据管线

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_workers` | 16 | DataLoader 工作进程数 |
| `--buffer_batches` | 20 | Shuffle 缓冲区大小（以 batch 为单位） |
| `--train_ratio` | 1.0 | 使用的训练 Row Group 比例 |
| `--valid_ratio` | 0.1 | 用于验证的 Row Group 比例 |
| `--seq_max_lens` | seq_a:256,seq_b:256,seq_c:512,seq_d:512 | 各序列域的最大截断长度 |

## NS 分组配置

`ns_groups.json` 定义了整数特征如何分组供 NS Tokenizer 使用。文件包含两个顶层 key：

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

值为 `feature_id`（即列名中的数字后缀，如 `user_int_feats_48`）。`train.py` 在运行时将其转换为 schema entry 索引。如果文件缺失，每个特征将作为独立的单元素分组。

**Token 数量公式**：`T = num_queries * num_sequences + num_ns`

以示例配置为例：`T = 1 * 4 + (7 + 1 + 4) = 16`。`d_model` 必须能被 `T` 整除（如 `d_model=64` 可行）。

## 模型检查点

每个检查点目录包含：

- `model.pt` —— 模型权重
- `schema.json` —— 特征 schema 副本（用于推理）
- `ns_groups.json` —— NS 分组配置副本（如提供）
- `train_config.json` —— 完整训练配置

## 许可

内部使用。
