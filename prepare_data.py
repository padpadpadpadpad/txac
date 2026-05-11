"""Prepare TAAC2026 demo data for PCVRHyFormer training.

Reads demo_1000.parquet, computes vocab sizes, and generates schema.json
so that dataset.py can load the data directly.
"""

import json
import os
import sys
import numpy as np
import pyarrow.parquet as pq


def main():
    parquet_path = "data_sample_1000/demo_1000.parquet"
    output_dir = "data_sample_1000"
    schema_out = os.path.join(output_dir, "schema.json")

    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found")
        sys.exit(1)

    pf = pq.ParquetFile(parquet_path)
    table = pf.read()
    df = table.to_pandas()

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ---- Helper: compute max value for vocab size ----
    def max_val(col_name):
        col = df[col_name]
        # For list columns, explode first
        if hasattr(col.iloc[0], '__len__') and not isinstance(col.iloc[0], str):
            all_vals = []
            for v in col:
                if v is not None:
                    all_vals.extend(v)
            arr = np.array(all_vals)
        else:
            arr = col.values
        arr = arr[arr > 0]  # ignore 0 and negative (padding/missing)
        return int(arr.max()) if len(arr) > 0 else 0

    # ---- user_int features ----
    user_int_cols = sorted([c for c in df.columns if c.startswith("user_int_feats_")],
                           key=lambda c: int(c.split("_")[-1]))
    user_int_schema = []
    for col_name in user_int_cols:
        fid = int(col_name.split("_")[-1])
        sample = df[col_name].iloc[0]
        if hasattr(sample, '__len__') and not isinstance(sample, str):
            # array column: compute max length and max value
            lengths = df[col_name].apply(lambda x: len(x) if x is not None else 0)
            dim = int(lengths.max())
            mv = max_val(col_name)
        else:
            dim = 1
            mv = max_val(col_name)
        vocab_size = mv + 1
        user_int_schema.append([fid, vocab_size, dim])
        print(f"  user_int_feats_{fid}: dim={dim}, vocab_size={vocab_size}")

    # ---- item_int features ----
    item_int_cols = sorted([c for c in df.columns if c.startswith("item_int_feats_")],
                           key=lambda c: int(c.split("_")[-1]))
    item_int_schema = []
    for col_name in item_int_cols:
        fid = int(col_name.split("_")[-1])
        sample = df[col_name].iloc[0]
        if hasattr(sample, '__len__') and not isinstance(sample, str):
            lengths = df[col_name].apply(lambda x: len(x) if x is not None else 0)
            dim = int(lengths.max())
            mv = max_val(col_name)
        else:
            dim = 1
            mv = max_val(col_name)
        vocab_size = mv + 1
        item_int_schema.append([fid, vocab_size, dim])
        print(f"  item_int_feats_{fid}: dim={dim}, vocab_size={vocab_size}")

    # ---- user_dense features ----
    user_dense_cols = sorted([c for c in df.columns if c.startswith("user_dense_feats_")],
                             key=lambda c: int(c.split("_")[-1]))
    user_dense_schema = []
    for col_name in user_dense_cols:
        fid = int(col_name.split("_")[-1])
        lengths = df[col_name].apply(lambda x: len(x) if x is not None else 0)
        dim = int(lengths.max())
        user_dense_schema.append([fid, dim])
        print(f"  user_dense_feats_{fid}: dim={dim}")

    # ---- Sequence domains ----
    # Map: seq_a -> domain_a, seq_b -> domain_b, seq_c -> domain_c, seq_d -> domain_d
    # ts_fid is the feature that contains per-position Unix timestamps
    domain_mapping = {
        "seq_a": ("domain_a", 39),
        "seq_b": ("domain_b", 67),
        "seq_c": ("domain_c", 27),
        "seq_d": ("domain_d", 26),
    }

    seq_schema = {}
    for seq_name, (domain_prefix, ts_fid) in domain_mapping.items():
        seq_cols = sorted(
            [c for c in df.columns if c.startswith(f"{domain_prefix}_seq_")],
            key=lambda c: int(c.split("_")[-1])
        )
        if not seq_cols:
            print(f"  WARNING: no columns found for {seq_name} ({domain_prefix}_seq_*)")
            continue

        features = []
        for col_name in seq_cols:
            fid = int(col_name.split("_")[-1])
            mv = max_val(col_name)
            vocab_size = mv + 1
            features.append([fid, vocab_size])

        seq_schema[seq_name] = {
            "prefix": f"{domain_prefix}_seq",
            "ts_fid": ts_fid,
            "features": features,
        }
        print(f"  {seq_name}: {len(features)} features, prefix={domain_prefix}_seq, ts_fid={ts_fid}")

    # ---- Build schema.json ----
    schema = {
        "user_int": user_int_schema,
        "item_int": item_int_schema,
        "user_dense": user_dense_schema,
        "item_dense": [],
        "seq": seq_schema,
    }

    with open(schema_out, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"\nSchema written to {schema_out}")
    print(f"  user_int: {len(user_int_schema)} features")
    print(f"  item_int: {len(item_int_schema)} features")
    print(f"  user_dense: {len(user_dense_schema)} features")
    print(f"  seq domains: {list(seq_schema.keys())}")


if __name__ == "__main__":
    main()
