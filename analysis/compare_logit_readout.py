#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_logit_readout.py

用途：
分析 concat_b2 与 WhiteningNet 在 test set 上的 classifier 读出行为，
重点关注关键工况上的：
- true class logit
- predicted class logit
- pred_minus_true
- top2 gap

核心目标：
1. 定量解释 WhiteningNet 在 S1725 上如何“纠正” prototype 结论
2. 定量解释 WhiteningNet 在 S1948 上如何整体翻向错误类
3. 为后续“读出稳定化 / 判别边界稳定化”方法设计提供依据

输入：
- concat_b2 test_embeddings.npz
- WhiteningNet test_embeddings.npz

输出：
- sample_logit_readout.csv
- condition_logit_readout_summary.csv
- key_condition_logit_readout_compare.csv
- key_condition_logit_readout_compare.txt

说明：
- 需要 npz 中存在 logits
- y_true 若不存在则报错
- y_pred 若不存在则由 argmax(logits) 自动推断
"""

import argparse
from pathlib import Path
from typing import List, Optional

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


# =========================
# 读取 test npz
# =========================

def load_test_npz(npz_path: Path, method_name: str):
    d = load_npz(npz_path)

    if method_name == "concat_b2":
        y_true_key = find_first_key(d, ["y", "y_true", "label"])
        y_pred_key = find_first_key(d, ["y_pred", "pred", "prediction"])
        logits_key = find_first_key(d, ["logits"])
        cond_key = find_first_key(d, ["condition_id", "condition", "cond"])
        sample_id_key = find_first_key(d, ["sample_id"])
    elif method_name == "WhiteningNet":
        y_true_key = find_first_key(d, ["y_true", "y", "label"])
        y_pred_key = find_first_key(d, ["y_pred", "pred", "prediction"])
        logits_key = find_first_key(d, ["logits"])
        cond_key = find_first_key(d, ["condition_id", "condition", "cond", "loader_name"])
        sample_id_key = find_first_key(d, ["sample_id"])
    else:
        raise ValueError(f"未知 method_name: {method_name}")

    if y_true_key is None:
        raise ValueError(f"[{method_name}] 未找到 y_true 字段。现有字段：{list(d.keys())}")
    if logits_key is None:
        raise ValueError(f"[{method_name}] 未找到 logits 字段。现有字段：{list(d.keys())}")

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

    if not (len(y_true) == len(y_pred) == len(condition_id) == len(sample_id) == logits.shape[0]):
        raise ValueError(
            f"[{method_name}] 长度不一致："
            f"y_true={len(y_true)}, y_pred={len(y_pred)}, condition_id={len(condition_id)}, "
            f"sample_id={len(sample_id)}, logits={logits.shape[0]}"
        )

    return y_true, y_pred, logits, condition_id, sample_id


# =========================
# 样本级分析
# =========================

def compute_sample_logit_table(
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: np.ndarray,
    condition_id: np.ndarray,
    sample_id: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for i in range(len(y_true)):
        yt = int(y_true[i])
        yp = int(y_pred[i])
        logit_vec = logits[i]

        if yt < 0 or yt >= len(logit_vec):
            raise ValueError(f"[{method}] y_true 越界：{yt}, logits_dim={len(logit_vec)}")
        if yp < 0 or yp >= len(logit_vec):
            raise ValueError(f"[{method}] y_pred 越界：{yp}, logits_dim={len(logit_vec)}")

        true_logit = float(logit_vec[yt])
        pred_logit = float(logit_vec[yp])
        pred_minus_true = float(pred_logit - true_logit)

        sorted_logits = np.sort(logit_vec)[::-1]
        top1 = float(sorted_logits[0])
        top2 = float(sorted_logits[1]) if len(sorted_logits) > 1 else float(sorted_logits[0])
        top2_gap = float(top1 - top2)

        rows.append(
            {
                "method": method,
                "sample_id": str(sample_id[i]),
                "condition_id": str(condition_id[i]),
                "y_true": yt,
                "y_pred": yp,
                "clf_correct": bool(yt == yp),
                "true_logit": true_logit,
                "pred_logit": pred_logit,
                "pred_minus_true": pred_minus_true,
                "top2_gap": top2_gap,
            }
        )

    return pd.DataFrame(rows)


# =========================
# 汇总
# =========================

def summarize_condition(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["method", "condition_id"], as_index=False).agg(
        n_samples=("y_true", "size"),
        clf_acc=("clf_correct", "mean"),
        true_logit_mean=("true_logit", "mean"),
        pred_logit_mean=("pred_logit", "mean"),
        pred_minus_true_mean=("pred_minus_true", "mean"),
        pred_minus_true_median=("pred_minus_true", "median"),
        pred_minus_true_std=("pred_minus_true", "std"),
        top2_gap_mean=("top2_gap", "mean"),
        top2_gap_median=("top2_gap", "median"),
    )
    return grouped


def add_major_prediction_modes(df_samples: pd.DataFrame, df_summary: pd.DataFrame) -> pd.DataFrame:
    extra_rows = []

    for _, row in df_summary.iterrows():
        method = row["method"]
        cond = row["condition_id"]
        sub = df_samples[(df_samples["method"] == method) & (df_samples["condition_id"] == cond)].copy()

        pred_counts = sub["y_pred"].value_counts()
        major_pred_class = int(pred_counts.idxmax()) if len(pred_counts) > 0 else None

        wrong = sub[sub["y_pred"] != sub["y_true"]].copy()
        if len(wrong) > 0:
            wrong_paths = wrong.apply(lambda r: f"{int(r['y_true'])}->{int(r['y_pred'])}", axis=1)
            major_wrong_path = wrong_paths.value_counts().idxmax()
        else:
            major_wrong_path = ""

        extra_rows.append(
            {
                "method": method,
                "condition_id": cond,
                "major_pred_class": major_pred_class,
                "major_wrong_path": major_wrong_path,
            }
        )

    df_extra = pd.DataFrame(extra_rows)
    out = df_summary.merge(df_extra, on=["method", "condition_id"], how="left")
    return out


# =========================
# 主函数
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Compare classifier logit readout for WhiteningNet and concat_b2")
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

    # concat_b2
    c_y_true, c_y_pred, c_logits, c_cond, c_sid = load_test_npz(Path(args.concat_test), "concat_b2")
    df_concat = compute_sample_logit_table(
        method="concat_b2",
        y_true=c_y_true,
        y_pred=c_y_pred,
        logits=c_logits,
        condition_id=c_cond,
        sample_id=c_sid,
    )

    # WhiteningNet
    w_y_true, w_y_pred, w_logits, w_cond, w_sid = load_test_npz(Path(args.whitening_test), "WhiteningNet")
    df_white = compute_sample_logit_table(
        method="WhiteningNet",
        y_true=w_y_true,
        y_pred=w_y_pred,
        logits=w_logits,
        condition_id=w_cond,
        sample_id=w_sid,
    )

    df_all = pd.concat([df_concat, df_white], axis=0, ignore_index=True)
    df_summary = summarize_condition(df_all)
    df_summary = add_major_prediction_modes(df_all, df_summary)

    key_conditions = args.key_conditions
    df_key = df_summary[df_summary["condition_id"].isin(key_conditions)].copy()
    df_key = df_key.sort_values(["condition_id", "method"]).reset_index(drop=True)

    # save
    df_all.to_csv(outdir / "sample_logit_readout.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(outdir / "condition_logit_readout_summary.csv", index=False, encoding="utf-8-sig")
    df_key.to_csv(outdir / "key_condition_logit_readout_compare.csv", index=False, encoding="utf-8-sig")

    with open(outdir / "key_condition_logit_readout_compare.txt", "w", encoding="utf-8") as f:
        f.write("Key Condition Logit Readout Compare\n")
        f.write("=" * 160 + "\n\n")
        if len(df_key) == 0:
            f.write("No matched key conditions found.\n")
        else:
            f.write(df_key.to_string(index=False))
            f.write("\n")

    # terminal
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 260)

    print("\n================ Condition Logit Readout Summary ================\n")
    print(df_summary.sort_values(["condition_id", "method"]).to_string(index=False))

    print("\n================ Key Condition Logit Readout Compare ================\n")
    if len(df_key) == 0:
        print("未匹配到关键工况。")
    else:
        print(df_key.to_string(index=False))

    print("\n输出完成：")
    print(f"  - {outdir / 'sample_logit_readout.csv'}")
    print(f"  - {outdir / 'condition_logit_readout_summary.csv'}")
    print(f"  - {outdir / 'key_condition_logit_readout_compare.csv'}")
    print(f"  - {outdir / 'key_condition_logit_readout_compare.txt'}")


if __name__ == "__main__":
    main()