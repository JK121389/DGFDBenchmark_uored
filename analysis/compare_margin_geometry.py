#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_margin_geometry.py

用途：
对比 WhiteningNet 与 concat_b2 在关键工况上的几何状态：
- d_true: 到真类原型距离
- d_wrong_min: 到最近错类原型距离
- margin = d_wrong_min - d_true

核心目标：
1. 判断 WhiteningNet 修复 S1725 是否写在表示几何里
2. 判断 WhiteningNet 毁坏 S1948 是否表现为几何翻转
3. 为后续第五章“更轻、更稳的类条件稳定机制”提供前置依据

输入：
A. concat_b2
   - train_embeddings.npz
   - test_embeddings.npz

B. WhiteningNet
   - test_embeddings.npz
   （由于当前没有 train embeddings，默认用 test 全集按类均值构造近似原型）

输出：
- sample_margin_compare.csv
- condition_margin_summary.csv
- key_condition_margin_compare.csv
- key_condition_margin_compare.txt

说明：
- concat_b2 默认优先读取字段 z / y / condition_id，若不存在会尝试 features / y_true
- WhiteningNet 默认读取 features / y_true / condition_id
- 只做分析，不修改原始结果文件
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# 基础工具
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def find_first_key(d: dict, candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


def normalize_str_array(arr) -> np.ndarray:
    arr = np.asarray(arr)
    return arr.astype(str)


def normalize_int_array(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind in {"U", "S", "O"}:
        return arr.astype(str).astype(int)
    return arr.astype(int)


def l2_distance_matrix(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    X: [N, D]
    P: [C, D]
    return: [N, C]
    """
    x2 = np.sum(X * X, axis=1, keepdims=True)
    p2 = np.sum(P * P, axis=1, keepdims=True).T
    xp = X @ P.T
    dist2 = np.maximum(x2 + p2 - 2 * xp, 0.0)
    return np.sqrt(dist2 + 1e-12)


def build_class_prototypes(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    按类均值构建原型
    返回：
    - class_ids: [C]
    - protos: [C, D]
    """
    class_ids = np.array(sorted(np.unique(labels)))
    protos = []
    for c in class_ids:
        mask = (labels == c)
        proto = features[mask].mean(axis=0)
        protos.append(proto)
    protos = np.stack(protos, axis=0)
    return class_ids, protos


def compute_margin_table(
    method: str,
    features: np.ndarray,
    labels: np.ndarray,
    condition_ids: np.ndarray,
    proto_class_ids: np.ndarray,
    proto_vectors: np.ndarray,
) -> pd.DataFrame:
    """
    对每个样本计算：
    - d_true
    - d_wrong_min
    - nearest_wrong_class
    - margin
    """
    dist_mat = l2_distance_matrix(features, proto_vectors)

    class_to_idx = {int(c): i for i, c in enumerate(proto_class_ids.tolist())}

    rows = []
    for i in range(features.shape[0]):
        y = int(labels[i])
        cond = str(condition_ids[i])

        if y not in class_to_idx:
            raise ValueError(f"[{method}] 标签 {y} 不在原型类集合中：{proto_class_ids.tolist()}")

        true_idx = class_to_idx[y]
        d_true = float(dist_mat[i, true_idx])

        wrong_mask = np.ones(len(proto_class_ids), dtype=bool)
        wrong_mask[true_idx] = False

        wrong_dists = dist_mat[i, wrong_mask]
        wrong_classes = proto_class_ids[wrong_mask]

        min_wrong_pos = int(np.argmin(wrong_dists))
        d_wrong_min = float(wrong_dists[min_wrong_pos])
        nearest_wrong_class = int(wrong_classes[min_wrong_pos])

        margin = float(d_wrong_min - d_true)

        rows.append(
            {
                "method": method,
                "condition_id": cond,
                "y_true": y,
                "d_true": d_true,
                "d_wrong_min": d_wrong_min,
                "nearest_wrong_class": nearest_wrong_class,
                "margin": margin,
                "margin_positive": bool(margin > 0),
            }
        )

    return pd.DataFrame(rows)


def summarize_condition_margins(df: pd.DataFrame) -> pd.DataFrame:
    """
    每个 method x condition 汇总：
    - n_samples
    - margin_mean / median
    - positive_margin_ratio
    - d_true_mean
    - d_wrong_min_mean
    """
    grouped = df.groupby(["method", "condition_id"], as_index=False).agg(
        n_samples=("margin", "size"),
        margin_mean=("margin", "mean"),
        margin_median=("margin", "median"),
        margin_std=("margin", "std"),
        positive_margin_ratio=("margin_positive", "mean"),
        d_true_mean=("d_true", "mean"),
        d_wrong_min_mean=("d_wrong_min", "mean"),
    )
    return grouped


# =========================
# 读取 concat_b2
# =========================

def load_concat_b2_train_prototypes(train_npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = load_npz(train_npz_path)

    feat_key = find_first_key(d, ["z", "features"])
    y_key = find_first_key(d, ["y", "y_true", "label"])
    if feat_key is None or y_key is None:
        raise ValueError(
            f"[concat_b2 train] 未找到特征或标签字段。现有字段：{list(d.keys())}"
        )

    feats = np.asarray(d[feat_key])
    labels = normalize_int_array(d[y_key])

    return build_class_prototypes(feats, labels)


def load_concat_b2_test_table(test_npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = load_npz(test_npz_path)

    feat_key = find_first_key(d, ["z", "features"])
    y_key = find_first_key(d, ["y", "y_true", "label"])
    cond_key = find_first_key(d, ["condition_id", "condition", "cond"])

    if feat_key is None or y_key is None or cond_key is None:
        raise ValueError(
            f"[concat_b2 test] 未找到特征/标签/condition_id 字段。现有字段：{list(d.keys())}"
        )

    feats = np.asarray(d[feat_key])
    labels = normalize_int_array(d[y_key])
    conds = normalize_str_array(d[cond_key])

    return feats, labels, conds


# =========================
# 读取 WhiteningNet
# =========================

def load_whitening_test_table_and_prototypes(test_npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = load_npz(test_npz_path)

    feat_key = find_first_key(d, ["features", "z"])
    y_key = find_first_key(d, ["y_true", "y", "label"])
    cond_key = find_first_key(d, ["condition_id", "condition", "cond", "loader_name"])

    if feat_key is None or y_key is None or cond_key is None:
        raise ValueError(
            f"[WhiteningNet test] 未找到特征/标签/condition_id 字段。现有字段：{list(d.keys())}"
        )

    feats = np.asarray(d[feat_key])
    labels = normalize_int_array(d[y_key])
    conds = normalize_str_array(d[cond_key])

    class_ids, protos = build_class_prototypes(feats, labels)
    return feats, labels, conds, class_ids, protos


# =========================
# 主逻辑
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Compare WhiteningNet vs concat_b2 margin geometry")
    parser.add_argument("--concat-train", type=str, required=True,
                        help="concat_b2 train_embeddings.npz")
    parser.add_argument("--concat-test", type=str, required=True,
                        help="concat_b2 test_embeddings.npz")
    parser.add_argument("--whitening-test", type=str, required=True,
                        help="WhiteningNet test_embeddings.npz")
    parser.add_argument("--outdir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--key-conditions", type=str, nargs="*",
                        default=["S1725_L400_T6", "S1948_L400_T11"],
                        help="关键工况列表")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    key_conditions = args.key_conditions

    # concat_b2
    concat_proto_class_ids, concat_protos = load_concat_b2_train_prototypes(Path(args.concat_train))
    concat_test_feats, concat_test_labels, concat_test_conds = load_concat_b2_test_table(Path(args.concat_test))
    df_concat = compute_margin_table(
        method="concat_b2",
        features=concat_test_feats,
        labels=concat_test_labels,
        condition_ids=concat_test_conds,
        proto_class_ids=concat_proto_class_ids,
        proto_vectors=concat_protos,
    )

    # WhiteningNet
    white_test_feats, white_test_labels, white_test_conds, white_proto_class_ids, white_protos = \
        load_whitening_test_table_and_prototypes(Path(args.whitening_test))
    df_white = compute_margin_table(
        method="WhiteningNet",
        features=white_test_feats,
        labels=white_test_labels,
        condition_ids=white_test_conds,
        proto_class_ids=white_proto_class_ids,
        proto_vectors=white_protos,
    )

    # merge
    df_all = pd.concat([df_concat, df_white], axis=0, ignore_index=True)
    df_summary = summarize_condition_margins(df_all)

    df_key = df_summary[df_summary["condition_id"].isin(key_conditions)].copy()
    df_key = df_key.sort_values(["condition_id", "method"]).reset_index(drop=True)

    # 输出
    df_all.to_csv(outdir / "sample_margin_compare.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(outdir / "condition_margin_summary.csv", index=False, encoding="utf-8-sig")
    df_key.to_csv(outdir / "key_condition_margin_compare.csv", index=False, encoding="utf-8-sig")

    with open(outdir / "key_condition_margin_compare.txt", "w", encoding="utf-8") as f:
        f.write("Key Condition Margin Compare\n")
        f.write("=" * 120 + "\n\n")
        if len(df_key) == 0:
            f.write("No matched key conditions found.\n")
        else:
            f.write(df_key.to_string(index=False))
            f.write("\n")

    # terminal
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 200)

    print("\n================ Condition Margin Summary ================\n")
    print(df_summary.sort_values(["condition_id", "method"]).to_string(index=False))

    print("\n================ Key Condition Margin Compare ================\n")
    if len(df_key) == 0:
        print("未匹配到关键工况。")
    else:
        print(df_key.to_string(index=False))

    print("\n输出完成：")
    print(f"  - {outdir / 'sample_margin_compare.csv'}")
    print(f"  - {outdir / 'condition_margin_summary.csv'}")
    print(f"  - {outdir / 'key_condition_margin_compare.csv'}")
    print(f"  - {outdir / 'key_condition_margin_compare.txt'}")


if __name__ == "__main__":
    main()


"""
python /root/py/multidiag_remote/DGFDBenchmark_uored/analysis/compare_margin_geometry.py \
  --concat-train /root/py/multidiag_remote/multidiag/outputs/2026-03-10/exp_align_supcon_proto_vce_concat_b2/run_001/artifacts/train_embeddings.npz \
  --concat-test /root/py/multidiag_remote/multidiag/outputs/2026-03-10/exp_align_supcon_proto_vce_concat_b2/run_001/artifacts/test_embeddings.npz \
  --whitening-test /root/py/multidiag_remote/DGFDBenchmark_uored/Output/WhiteningNet_UORED/whiteningnet_uored_b001_allsrc_alltgt__embed_20260312_160749/artifacts/test_embeddings.npz \
  --outdir /root/py/multidiag_remote/analysis_outputs/compare_margin_geometry

/root/py/multidiag_remote/analysis_outputs/compare_margin_geometry/key_condition_margin_compare.csv
"""