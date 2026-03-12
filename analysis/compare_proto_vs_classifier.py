#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_proto_vs_classifier.py

用途：
在统一 train-prototype 口径下，对比 WhiteningNet 与 concat_b2 的：
1. nearest-prototype class
2. classifier predicted class
3. true class

核心目标：
- 判断 WhiteningNet / concat_b2 在关键工况上的错误，主要写在 prototype 几何还是 classifier 读出
- 特别分析：
  - S1725: WhiteningNet 为什么几乎全对，而 train-prototype margin 却不支持？
  - S1948: WhiteningNet 为什么 classifier 全错，而 train-prototype margin 看起来健康？

输入：
A. concat_b2
   - train_embeddings.npz
   - test_embeddings.npz

B. WhiteningNet
   - train_embeddings.npz
   - test_embeddings.npz

输出：
- sample_proto_vs_classifier.csv
- condition_proto_classifier_summary.csv
- key_condition_proto_classifier_compare.csv
- key_condition_proto_classifier_compare.txt

说明：
- train prototype 统一由 train_embeddings 按类均值构造
- classifier prediction 优先从 test_embeddings.npz 中读取 y_pred
- 若缺 y_pred，但存在 logits，则自动用 argmax(logits) 计算 classifier prediction
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

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
    return np.asarray(arr).astype(str)


def normalize_int_array(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind in {"U", "S", "O"}:
        return arr.astype(str).astype(int)
    return arr.astype(int)


def l2_distance_matrix(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    p2 = np.sum(P * P, axis=1, keepdims=True).T
    xp = X @ P.T
    dist2 = np.maximum(x2 + p2 - 2 * xp, 0.0)
    return np.sqrt(dist2 + 1e-12)


def build_class_prototypes(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    class_ids = np.array(sorted(np.unique(labels)))
    protos = []
    for c in class_ids:
        mask = (labels == c)
        proto = features[mask].mean(axis=0)
        protos.append(proto)
    protos = np.stack(protos, axis=0)
    return class_ids, protos


def infer_y_pred_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"logits 维度应为 [N, C]，当前为 {logits.shape}")
    return np.argmax(logits, axis=1).astype(int)


# =========================
# 通用 embeddings 读取
# =========================

def load_method_npz(npz_path: Path, method_name: str):
    d = load_npz(npz_path)

    if method_name == "concat_b2":
        feat_key = find_first_key(d, ["z", "features"])
        y_true_key = find_first_key(d, ["y", "y_true", "label"])
        y_pred_key = find_first_key(d, ["y_pred", "pred", "prediction"])
        logits_key = find_first_key(d, ["logits"])
        cond_key = find_first_key(d, ["condition_id", "condition", "cond"])
        sample_id_key = find_first_key(d, ["sample_id"])
    elif method_name == "WhiteningNet":
        feat_key = find_first_key(d, ["features", "z"])
        y_true_key = find_first_key(d, ["y_true", "y", "label"])
        y_pred_key = find_first_key(d, ["y_pred", "pred", "prediction"])
        logits_key = find_first_key(d, ["logits"])
        cond_key = find_first_key(d, ["condition_id", "condition", "cond", "loader_name"])
        sample_id_key = find_first_key(d, ["sample_id"])
    else:
        raise ValueError(f"未知 method_name: {method_name}")

    if feat_key is None:
        raise ValueError(f"[{method_name}] 未找到 feature 字段。现有字段：{list(d.keys())}")
    if y_true_key is None:
        raise ValueError(f"[{method_name}] 未找到 y_true 字段。现有字段：{list(d.keys())}")

    features = np.asarray(d[feat_key])
    y_true = normalize_int_array(d[y_true_key])

    # classifier prediction：优先读 y_pred；没有就从 logits 推
    if y_pred_key is not None:
        y_pred = normalize_int_array(d[y_pred_key])
    else:
        if logits_key is None:
            raise ValueError(
                f"[{method_name}] 未找到 y_pred，也未找到 logits。现有字段：{list(d.keys())}"
            )
        y_pred = infer_y_pred_from_logits(np.asarray(d[logits_key]))

    if cond_key is not None:
        condition_id = normalize_str_array(d[cond_key])
    else:
        condition_id = np.array(["unknown"] * len(y_true), dtype=str)

    if sample_id_key is not None:
        sample_id = normalize_str_array(d[sample_id_key])
    else:
        sample_id = np.array([str(i) for i in range(len(y_true))], dtype=str)

    if not (len(features) == len(y_true) == len(y_pred) == len(condition_id) == len(sample_id)):
        raise ValueError(
            f"[{method_name}] 特征/标签/预测/condition/sample_id 长度不一致："
            f"{len(features)}, {len(y_true)}, {len(y_pred)}, {len(condition_id)}, {len(sample_id)}"
        )

    return features, y_true, y_pred, condition_id, sample_id


# =========================
# 主体分析
# =========================

def compute_proto_vs_classifier_table(
    method: str,
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    test_y_true: np.ndarray,
    test_y_pred: np.ndarray,
    test_condition_id: np.ndarray,
    test_sample_id: np.ndarray,
) -> pd.DataFrame:
    proto_class_ids, proto_vectors = build_class_prototypes(train_feats, train_labels)
    dist_mat = l2_distance_matrix(test_feats, proto_vectors)
    nearest_proto_idx = np.argmin(dist_mat, axis=1)
    nearest_proto_class = proto_class_ids[nearest_proto_idx]

    rows = []
    for i in range(len(test_y_true)):
        y_true = int(test_y_true[i])
        y_pred = int(test_y_pred[i])
        y_proto = int(nearest_proto_class[i])

        rows.append(
            {
                "method": method,
                "sample_id": str(test_sample_id[i]),
                "condition_id": str(test_condition_id[i]),
                "y_true": y_true,
                "y_pred": y_pred,
                "y_proto": y_proto,
                "proto_correct": bool(y_proto == y_true),
                "clf_correct": bool(y_pred == y_true),
                "proto_clf_agree": bool(y_proto == y_pred),
                "proto_correct_clf_wrong": bool((y_proto == y_true) and (y_pred != y_true)),
                "proto_wrong_clf_correct": bool((y_proto != y_true) and (y_pred == y_true)),
                "both_wrong_same": bool((y_proto != y_true) and (y_pred != y_true) and (y_proto == y_pred)),
                "both_wrong_diff": bool((y_proto != y_true) and (y_pred != y_true) and (y_proto != y_pred)),
            }
        )

    return pd.DataFrame(rows)


def summarize_condition(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["method", "condition_id"], as_index=False).agg(
        n_samples=("y_true", "size"),
        proto_acc=("proto_correct", "mean"),
        clf_acc=("clf_correct", "mean"),
        proto_clf_agreement=("proto_clf_agree", "mean"),
        proto_correct_clf_wrong_ratio=("proto_correct_clf_wrong", "mean"),
        proto_wrong_clf_correct_ratio=("proto_wrong_clf_correct", "mean"),
        both_wrong_same_ratio=("both_wrong_same", "mean"),
        both_wrong_diff_ratio=("both_wrong_diff", "mean"),
    )
    return grouped


def add_major_modes(df_samples: pd.DataFrame, df_summary: pd.DataFrame) -> pd.DataFrame:
    extra_rows = []

    for _, row in df_summary.iterrows():
        method = row["method"]
        cond = row["condition_id"]
        sub = df_samples[(df_samples["method"] == method) & (df_samples["condition_id"] == cond)].copy()

        proto_wrong = sub[sub["y_proto"] != sub["y_true"]]
        clf_wrong = sub[sub["y_pred"] != sub["y_true"]]

        if len(proto_wrong) > 0:
            proto_pairs = proto_wrong.apply(lambda r: f"{int(r['y_true'])}->{int(r['y_proto'])}", axis=1)
            major_proto_error_path = proto_pairs.value_counts().idxmax()
        else:
            major_proto_error_path = ""

        if len(clf_wrong) > 0:
            clf_pairs = clf_wrong.apply(lambda r: f"{int(r['y_true'])}->{int(r['y_pred'])}", axis=1)
            major_clf_error_path = clf_pairs.value_counts().idxmax()
        else:
            major_clf_error_path = ""

        extra_rows.append(
            {
                "method": method,
                "condition_id": cond,
                "major_proto_error_path": major_proto_error_path,
                "major_clf_error_path": major_clf_error_path,
            }
        )

    df_extra = pd.DataFrame(extra_rows)
    out = df_summary.merge(df_extra, on=["method", "condition_id"], how="left")
    return out


# =========================
# 命令行
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare nearest-prototype class vs classifier predicted class for WhiteningNet and concat_b2"
    )
    parser.add_argument("--concat-train", type=str, required=True,
                        help="concat_b2 train_embeddings.npz")
    parser.add_argument("--concat-test", type=str, required=True,
                        help="concat_b2 test_embeddings.npz")
    parser.add_argument("--whitening-train", type=str, required=True,
                        help="WhiteningNet train_embeddings.npz")
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

    # concat_b2
    c_train_feats, c_train_y, _, _, _ = load_method_npz(Path(args.concat_train), "concat_b2")
    c_test_feats, c_test_y_true, c_test_y_pred, c_test_cond, c_test_sid = load_method_npz(Path(args.concat_test), "concat_b2")

    df_concat = compute_proto_vs_classifier_table(
        method="concat_b2",
        train_feats=c_train_feats,
        train_labels=c_train_y,
        test_feats=c_test_feats,
        test_y_true=c_test_y_true,
        test_y_pred=c_test_y_pred,
        test_condition_id=c_test_cond,
        test_sample_id=c_test_sid,
    )

    # WhiteningNet
    w_train_feats, w_train_y, _, _, _ = load_method_npz(Path(args.whitening_train), "WhiteningNet")
    w_test_feats, w_test_y_true, w_test_y_pred, w_test_cond, w_test_sid = load_method_npz(Path(args.whitening_test), "WhiteningNet")

    df_white = compute_proto_vs_classifier_table(
        method="WhiteningNet",
        train_feats=w_train_feats,
        train_labels=w_train_y,
        test_feats=w_test_feats,
        test_y_true=w_test_y_true,
        test_y_pred=w_test_y_pred,
        test_condition_id=w_test_cond,
        test_sample_id=w_test_sid,
    )

    df_all = pd.concat([df_concat, df_white], axis=0, ignore_index=True)
    df_summary = summarize_condition(df_all)
    df_summary = add_major_modes(df_all, df_summary)

    key_conditions = args.key_conditions
    df_key = df_summary[df_summary["condition_id"].isin(key_conditions)].copy()
    df_key = df_key.sort_values(["condition_id", "method"]).reset_index(drop=True)

    # 保存
    df_all.to_csv(outdir / "sample_proto_vs_classifier.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(outdir / "condition_proto_classifier_summary.csv", index=False, encoding="utf-8-sig")
    df_key.to_csv(outdir / "key_condition_proto_classifier_compare.csv", index=False, encoding="utf-8-sig")

    with open(outdir / "key_condition_proto_classifier_compare.txt", "w", encoding="utf-8") as f:
        f.write("Key Condition Prototype vs Classifier Compare\n")
        f.write("=" * 160 + "\n\n")
        if len(df_key) == 0:
            f.write("No matched key conditions found.\n")
        else:
            f.write(df_key.to_string(index=False))
            f.write("\n")

    # 终端输出
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 260)

    print("\n================ Condition Proto vs Classifier Summary ================\n")
    print(df_summary.sort_values(["condition_id", "method"]).to_string(index=False))

    print("\n================ Key Condition Proto vs Classifier Compare ================\n")
    if len(df_key) == 0:
        print("未匹配到关键工况。")
    else:
        print(df_key.to_string(index=False))

    print("\n输出完成：")
    print(f"  - {outdir / 'sample_proto_vs_classifier.csv'}")
    print(f"  - {outdir / 'condition_proto_classifier_summary.csv'}")
    print(f"  - {outdir / 'key_condition_proto_classifier_compare.csv'}")
    print(f"  - {outdir / 'key_condition_proto_classifier_compare.txt'}")


if __name__ == "__main__":
    main()

"""
python /root/py/multidiag_remote/DGFDBenchmark_uored/analysis/compare_proto_vs_classifier.py   
    --concat-train /root/py/multidiag_remote/multidiag/outputs/2026-03-10/exp_align_supcon_proto_vce_concat_b2/run_001/artifacts/train_embeddings.npz   
    --concat-test /root/py/multidiag_remote/multidiag/outputs/2026-03-10/exp_align_supcon_proto_vce_concat_b2/run_001/artifacts/test_embeddings.npz   
    --whitening-train /root/py/multidiag_remote/DGFDBenchmark_uored/Output/WhiteningNet_UORED/whiteningnet_uored_b001_allsrc_alltgt__embed_20260312_170648/artifacts/train_embeddings.npz   
    --whitening-test /root/py/multidiag_remote/DGFDBenchmark_uored/Output/WhiteningNet_UORED/whiteningnet_uored_b001_allsrc_alltgt__embed_20260312_170648/artifacts/test_embeddings.npz   
    --outdir /root/py/multidiag_remote/analysis_outputs/compare_proto_vs_classifier

"""