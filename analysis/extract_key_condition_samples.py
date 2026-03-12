#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_key_condition_samples.py

用途：
抽取 concat_b2 与 WhiteningNet 在关键工况上的 sample-level 对照信息，便于人工 spot check。

输出字段包括：
- sample_id
- condition_id
- y_true
- y_proto
- y_pred
- proto_correct
- clf_correct
- proto_clf_agree
- true_logit
- pred_logit
- pred_minus_true
- top2_gap

输入：
A. concat_b2
   - train_embeddings.npz
   - test_embeddings.npz

B. WhiteningNet
   - train_embeddings.npz
   - test_embeddings.npz

输出：
- key_condition_samples_concat_b2.csv
- key_condition_samples_WhiteningNet.csv
- key_condition_mode_summary.csv
- key_condition_mode_summary.txt

说明：
- train prototype 统一由 train_embeddings 按类均值构造
- y_pred 优先读取；若不存在则由 logits.argmax(axis=1) 生成
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


def infer_y_pred_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"logits 维度应为 [N, C]，当前为 {logits.shape}")
    return np.argmax(logits, axis=1).astype(int)


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


# =========================
# 读取 npz
# =========================

def load_train_npz(npz_path: Path, method_name: str):
    d = load_npz(npz_path)

    if method_name == "concat_b2":
        feat_key = find_first_key(d, ["z", "features"])
        y_key = find_first_key(d, ["y", "y_true", "label"])
    elif method_name == "WhiteningNet":
        feat_key = find_first_key(d, ["features", "z"])
        y_key = find_first_key(d, ["y_true", "y", "label"])
    else:
        raise ValueError(f"未知 method_name: {method_name}")

    if feat_key is None or y_key is None:
        raise ValueError(f"[{method_name} train] 缺少必要字段。现有字段：{list(d.keys())}")

    features = np.asarray(d[feat_key])
    labels = normalize_int_array(d[y_key])
    return features, labels


def load_test_npz(npz_path: Path, method_name: str):
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

    if feat_key is None or y_true_key is None or logits_key is None:
        raise ValueError(f"[{method_name} test] 缺少必要字段。现有字段：{list(d.keys())}")

    features = np.asarray(d[feat_key])
    y_true = normalize_int_array(d[y_true_key])
    logits = np.asarray(d[logits_key])

    if y_pred_key is not None:
        y_pred = normalize_int_array(d[y_pred_key])
    else:
        y_pred = infer_y_pred_from_logits(logits)

    if cond_key is not None:
        condition_id = normalize_str_array(d[cond_key])
    else:
        condition_id = np.array(["unknown"] * len(y_true), dtype=str)

    if sample_id_key is not None:
        sample_id = normalize_str_array(d[sample_id_key])
    else:
        sample_id = np.array([str(i) for i in range(len(y_true))], dtype=str)

    if not (len(features) == len(y_true) == len(y_pred) == len(condition_id) == len(sample_id) == logits.shape[0]):
        raise ValueError(
            f"[{method_name}] 长度不一致："
            f"features={len(features)}, y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"condition_id={len(condition_id)}, sample_id={len(sample_id)}, logits={logits.shape[0]}"
        )

    return features, y_true, y_pred, logits, condition_id, sample_id


# =========================
# 样本级构表
# =========================

def build_sample_table(
    method: str,
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    test_y_true: np.ndarray,
    test_y_pred: np.ndarray,
    test_logits: np.ndarray,
    test_condition_id: np.ndarray,
    test_sample_id: np.ndarray,
    key_conditions: List[str],
) -> pd.DataFrame:
    proto_class_ids, proto_vectors = build_class_prototypes(train_feats, train_labels)
    dist_mat = l2_distance_matrix(test_feats, proto_vectors)
    nearest_proto_idx = np.argmin(dist_mat, axis=1)
    nearest_proto_class = proto_class_ids[nearest_proto_idx]

    rows = []
    for i in range(len(test_y_true)):
        cond = str(test_condition_id[i])
        if cond not in key_conditions:
            continue

        yt = int(test_y_true[i])
        yp = int(test_y_pred[i])
        yproto = int(nearest_proto_class[i])
        logits_i = test_logits[i]

        true_logit = float(logits_i[yt])
        pred_logit = float(logits_i[yp])
        pred_minus_true = float(pred_logit - true_logit)

        sorted_logits = np.sort(logits_i)[::-1]
        top1 = float(sorted_logits[0])
        top2 = float(sorted_logits[1]) if len(sorted_logits) > 1 else float(sorted_logits[0])
        top2_gap = float(top1 - top2)

        rows.append(
            {
                "method": method,
                "sample_id": str(test_sample_id[i]),
                "condition_id": cond,
                "y_true": yt,
                "y_proto": yproto,
                "y_pred": yp,
                "proto_correct": bool(yproto == yt),
                "clf_correct": bool(yp == yt),
                "proto_clf_agree": bool(yproto == yp),
                "proto_correct_clf_wrong": bool((yproto == yt) and (yp != yt)),
                "proto_wrong_clf_correct": bool((yproto != yt) and (yp == yt)),
                "both_wrong_same": bool((yproto != yt) and (yp != yt) and (yproto == yp)),
                "both_wrong_diff": bool((yproto != yt) and (yp != yt) and (yproto != yp)),
                "true_logit": true_logit,
                "pred_logit": pred_logit,
                "pred_minus_true": pred_minus_true,
                "top2_gap": top2_gap,
            }
        )

    return pd.DataFrame(rows)


def summarize_modes(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["method", "condition_id"], as_index=False).agg(
        n_samples=("y_true", "size"),
        proto_acc=("proto_correct", "mean"),
        clf_acc=("clf_correct", "mean"),
        proto_clf_agreement=("proto_clf_agree", "mean"),
        proto_correct_clf_wrong_ratio=("proto_correct_clf_wrong", "mean"),
        proto_wrong_clf_correct_ratio=("proto_wrong_clf_correct", "mean"),
        both_wrong_same_ratio=("both_wrong_same", "mean"),
        both_wrong_diff_ratio=("both_wrong_diff", "mean"),
        true_logit_mean=("true_logit", "mean"),
        pred_logit_mean=("pred_logit", "mean"),
        pred_minus_true_mean=("pred_minus_true", "mean"),
        top2_gap_mean=("top2_gap", "mean"),
    )
    return grouped


# =========================
# 主函数
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Extract key-condition sample-level comparison for concat_b2 and WhiteningNet")
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

    key_conditions = args.key_conditions

    # concat_b2
    c_train_feats, c_train_labels = load_train_npz(Path(args.concat_train), "concat_b2")
    c_test_feats, c_test_y_true, c_test_y_pred, c_test_logits, c_test_cond, c_test_sid = load_test_npz(
        Path(args.concat_test), "concat_b2"
    )
    df_concat = build_sample_table(
        method="concat_b2",
        train_feats=c_train_feats,
        train_labels=c_train_labels,
        test_feats=c_test_feats,
        test_y_true=c_test_y_true,
        test_y_pred=c_test_y_pred,
        test_logits=c_test_logits,
        test_condition_id=c_test_cond,
        test_sample_id=c_test_sid,
        key_conditions=key_conditions,
    )

    # WhiteningNet
    w_train_feats, w_train_labels = load_train_npz(Path(args.whitening_train), "WhiteningNet")
    w_test_feats, w_test_y_true, w_test_y_pred, w_test_logits, w_test_cond, w_test_sid = load_test_npz(
        Path(args.whitening_test), "WhiteningNet"
    )
    df_white = build_sample_table(
        method="WhiteningNet",
        train_feats=w_train_feats,
        train_labels=w_train_labels,
        test_feats=w_test_feats,
        test_y_true=w_test_y_true,
        test_y_pred=w_test_y_pred,
        test_logits=w_test_logits,
        test_condition_id=w_test_cond,
        test_sample_id=w_test_sid,
        key_conditions=key_conditions,
    )

    df_all = pd.concat([df_concat, df_white], axis=0, ignore_index=True)
    df_summary = summarize_modes(df_all).sort_values(["condition_id", "method"]).reset_index(drop=True)

    # 保存
    df_concat.to_csv(outdir / "key_condition_samples_concat_b2.csv", index=False, encoding="utf-8-sig")
    df_white.to_csv(outdir / "key_condition_samples_WhiteningNet.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(outdir / "key_condition_mode_summary.csv", index=False, encoding="utf-8-sig")

    with open(outdir / "key_condition_mode_summary.txt", "w", encoding="utf-8") as f:
        f.write("Key Condition Sample-Level Mode Summary\n")
        f.write("=" * 160 + "\n\n")
        if len(df_summary) == 0:
            f.write("No matched key conditions found.\n")
        else:
            f.write(df_summary.to_string(index=False))
            f.write("\n")

    # 终端输出
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 260)

    print("\n================ Key Condition Sample-Level Mode Summary ================\n")
    if len(df_summary) == 0:
        print("未匹配到关键工况。")
    else:
        print(df_summary.to_string(index=False))

    print("\n输出完成：")
    print(f"  - {outdir / 'key_condition_samples_concat_b2.csv'}")
    print(f"  - {outdir / 'key_condition_samples_WhiteningNet.csv'}")
    print(f"  - {outdir / 'key_condition_mode_summary.csv'}")
    print(f"  - {outdir / 'key_condition_mode_summary.txt'}")


if __name__ == "__main__":
    main()