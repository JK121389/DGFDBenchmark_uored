#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_condition_behavior.py

用途：
统一比较四组实验（内部 concat_b2 + 外部 CNNC / CCDG / WhiteningNet）
在各 test condition 上的行为差异，重点输出：
1. 每个方法、每个 condition 的 acc / n_correct / n_wrong
2. 主错通道（如 0->3、2->0）
3. 主错通道集中度
4. 关键工况（默认 S1725_L400_T6 / S1948_L400_T11）对照表

支持两类输入：
A. 内部实验（如 concat_b2）
   - 单个总 test_preds.csv，需按 condition_id 自动拆分
B. 外部实验（如 CNNC / CCDG / WhiteningNet）
   - artifacts/test_preds__<condition>.csv
   - 可选 analysis_condition__<condition>/condition_summary.json

输出：
- method_condition_behavior_summary.csv
- key_condition_compare.csv

示例：
python compare_condition_behavior.py \
  --concat-b2 /root/py/multidiag_remote/multidiag/outputs/2026-03-10/exp_align_supcon_proto_vce_concat_b2/run_001 \
  --cnnc /root/py/multidiag_remote/DGFDBenchmark_uored/Output/CNNC_UORED/cnnc_uored_b001_allsrc_alltgt \
  --ccdg /root/py/multidiag_remote/DGFDBenchmark_uored/Output/CCDG_UORED/ccdg_uored_b001_allsrc_alltgt \
  --whitening /root/py/multidiag_remote/DGFDBenchmark_uored/Output/WhiteningNet_UORED/whiteningnet_uored_b001_allsrc_alltgt \
  --outdir /root/py/multidiag_remote/analysis_outputs/compare_condition_behavior

说明：
- 该脚本不依赖 condition_summary.json 的具体字段存在；若字段不匹配，会自动退回用 csv 自己统计。
- 该脚本优先以 test_preds.csv 的逐样本统计为准。
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =========================
# 可兼容的列名候选
# =========================

TRUE_COL_CANDIDATES = [
    "y_true", "label", "target", "y", "gt", "true_label", "true"
]

PRED_COL_CANDIDATES = [
    "y_pred", "pred", "prediction", "pred_label", "preds", "hat_y"
]

COND_COL_CANDIDATES = [
    "condition_id", "condition", "cond", "domain", "target_condition"
]


# =========================
# 基础工具函数
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_condition_from_filename(filename: str) -> Optional[str]:
    """
    从 test_preds__S1709_L0_T29.csv 提取 condition_id
    """
    m = re.match(r"test_preds__(.+)\.csv$", filename)
    if m:
        return m.group(1)
    return None


def infer_internal_test_preds_path(run_dir: Path) -> Path:
    """
    内部 concat_b2 的 test_preds.csv 可能在：
    - artifacts/test_preds.csv
    - predictions/test_preds.csv
    优先 artifacts
    """
    p1 = run_dir / "artifacts" / "test_preds.csv"
    p2 = run_dir / "predictions" / "test_preds.csv"

    if p1.exists():
        return p1
    if p2.exists():
        return p2

    raise FileNotFoundError(
        f"未找到内部实验 test_preds.csv，已检查：\n  {p1}\n  {p2}"
    )


def normalize_label_series(s: pd.Series) -> pd.Series:
    """
    尽量把标签转成更稳定的字符串形式，避免 int/float/string 混乱。
    """
    return s.astype(str).str.strip()


def compute_behavior_from_df(
    df: pd.DataFrame,
    method_name: str,
    condition_id: str,
) -> Dict[str, object]:
    """
    从逐样本预测表中，统计 condition 级行为摘要。
    """
    true_col = find_first_existing(df, TRUE_COL_CANDIDATES)
    pred_col = find_first_existing(df, PRED_COL_CANDIDATES)

    if true_col is None:
        raise ValueError(
            f"[{method_name} | {condition_id}] 未找到真实标签列，可选列名：{TRUE_COL_CANDIDATES}\n"
            f"当前列：{list(df.columns)}"
        )
    if pred_col is None:
        raise ValueError(
            f"[{method_name} | {condition_id}] 未找到预测标签列，可选列名：{PRED_COL_CANDIDATES}\n"
            f"当前列：{list(df.columns)}"
        )

    y_true = normalize_label_series(df[true_col])
    y_pred = normalize_label_series(df[pred_col])

    n_samples = len(df)
    correct_mask = (y_true == y_pred)
    n_correct = int(correct_mask.sum())
    n_wrong = int((~correct_mask).sum())
    acc = float(n_correct / n_samples) if n_samples > 0 else 0.0

    # 统计错误通道
    error_pairs = list(zip(y_true[~correct_mask], y_pred[~correct_mask]))
    error_counter = Counter(error_pairs)

    if n_wrong > 0:
        (major_true, major_pred), major_error_count = error_counter.most_common(1)[0]
        major_error_path = f"{major_true}->{major_pred}"
        major_error_ratio_among_wrong = float(major_error_count / n_wrong)
    else:
        major_error_path = ""
        major_error_count = 0
        major_error_ratio_among_wrong = 0.0

    # 近似判定是否“几乎单通道塌陷”
    # 这里不死板要求 410/410，只看错误是否高度集中到一个通道
    # 你后续可在论文里按需要再人工解释。
    all_wrong_single_channel = (
        (n_wrong == n_samples and n_samples > 0 and major_error_count == n_wrong)
        or (n_wrong > 0 and major_error_ratio_among_wrong >= 0.95)
    )

    return {
        "method": method_name,
        "condition_id": condition_id,
        "n_samples": n_samples,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "acc": acc,
        "major_error_path": major_error_path,
        "major_error_count": major_error_count,
        "major_error_ratio_among_wrong": major_error_ratio_among_wrong,
        "all_wrong_single_channel": bool(all_wrong_single_channel),
    }


def parse_external_condition_summary(summary_json: Optional[dict]) -> Dict[str, object]:
    """
    尝试从外部 condition_summary.json 中抽取可能有帮助的字段。
    不强依赖，提取不到就返回空字段。
    """
    if not summary_json:
        return {}

    out = {}

    # 常见信息尽量宽松提取
    for key in [
        "condition_id", "acc", "accuracy", "n_samples", "n_correct", "n_wrong",
        "major_error_path", "major_error_count", "major_error_ratio_among_wrong"
    ]:
        if key in summary_json:
            out[f"summary_{key}"] = summary_json[key]

    return out


# =========================
# 两类实验读取逻辑
# =========================

def collect_internal_concat_b2(run_dir: Path, method_name: str = "concat_b2") -> List[Dict[str, object]]:
    """
    内部实验：单个总 test_preds.csv，需要按 condition_id 分组统计。
    """
    test_preds_path = infer_internal_test_preds_path(run_dir)
    df = safe_read_csv(test_preds_path)

    cond_col = find_first_existing(df, COND_COL_CANDIDATES)
    if cond_col is None:
        raise ValueError(
            f"[{method_name}] 未找到 condition_id 列，可选列名：{COND_COL_CANDIDATES}\n"
            f"当前列：{list(df.columns)}"
        )

    rows = []
    for condition_id, sub_df in df.groupby(cond_col):
        condition_id = str(condition_id)
        row = compute_behavior_from_df(sub_df, method_name=method_name, condition_id=condition_id)
        rows.append(row)

    rows.sort(key=lambda x: x["condition_id"])
    return rows


def collect_external_method(exp_dir: Path, method_name: str) -> List[Dict[str, object]]:
    """
    外部实验：artifacts/test_preds__<condition>.csv
    """
    artifacts_dir = exp_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"[{method_name}] 未找到 artifacts 目录：{artifacts_dir}")

    rows = []
    csv_files = sorted(artifacts_dir.glob("test_preds__*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"[{method_name}] 未找到 test_preds__*.csv：{artifacts_dir}")

    for csv_path in csv_files:
        condition_id = extract_condition_from_filename(csv_path.name)
        if condition_id is None:
            continue

        df = safe_read_csv(csv_path)
        row = compute_behavior_from_df(df, method_name=method_name, condition_id=condition_id)

        # 尝试补充对应的 summary 信息
        summary_json_path = exp_dir / f"analysis_condition__{condition_id}" / "condition_summary.json"
        summary_json = load_json_if_exists(summary_json_path)
        row.update(parse_external_condition_summary(summary_json))

        rows.append(row)

    rows.sort(key=lambda x: x["condition_id"])
    return rows


# =========================
# 汇总与输出
# =========================

def build_dataframe(all_rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)

    desired_cols = [
        "method",
        "condition_id",
        "n_samples",
        "n_correct",
        "n_wrong",
        "acc",
        "major_error_path",
        "major_error_count",
        "major_error_ratio_among_wrong",
        "all_wrong_single_channel",
        # 外部 summary 补充字段（若存在）
        "summary_condition_id",
        "summary_acc",
        "summary_accuracy",
        "summary_n_samples",
        "summary_n_correct",
        "summary_n_wrong",
        "summary_major_error_path",
        "summary_major_error_count",
        "summary_major_error_ratio_among_wrong",
    ]

    existing_cols = [c for c in desired_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]

    return df[existing_cols + other_cols]


def save_outputs(
    df_all: pd.DataFrame,
    outdir: Path,
    key_conditions: List[str],
) -> None:
    ensure_dir(outdir)

    all_csv = outdir / "method_condition_behavior_summary.csv"
    key_csv = outdir / "key_condition_compare.csv"

    df_all_sorted = df_all.sort_values(["condition_id", "method"]).reset_index(drop=True)
    df_all_sorted.to_csv(all_csv, index=False, encoding="utf-8-sig")

    df_key = df_all_sorted[df_all_sorted["condition_id"].isin(key_conditions)].copy()
    df_key.to_csv(key_csv, index=False, encoding="utf-8-sig")

    preview_txt = outdir / "key_condition_compare.txt"
    with open(preview_txt, "w", encoding="utf-8") as f:
        f.write("Key Condition Compare\n")
        f.write("=" * 80 + "\n\n")
        if len(df_key) == 0:
            f.write("No matched key conditions found.\n")
        else:
            f.write(df_key.to_string(index=False))
            f.write("\n")


def print_terminal_summary(df_all: pd.DataFrame, key_conditions: List[str]) -> None:
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 200)

    print("\n================ 全部 condition 行为摘要 ================\n")
    print(
        df_all.sort_values(["condition_id", "method"])[
            [
                "method", "condition_id", "n_samples", "acc",
                "major_error_path", "major_error_count",
                "major_error_ratio_among_wrong", "all_wrong_single_channel"
            ]
        ].to_string(index=False)
    )

    df_key = df_all[df_all["condition_id"].isin(key_conditions)].copy()
    print("\n================ 关键工况对照 ================\n")
    if len(df_key) == 0:
        print("未匹配到关键工况。")
    else:
        print(
            df_key.sort_values(["condition_id", "method"])[
                [
                    "method", "condition_id", "n_samples", "acc",
                    "major_error_path", "major_error_count",
                    "major_error_ratio_among_wrong", "all_wrong_single_channel"
                ]
            ].to_string(index=False)
        )


# =========================
# 命令行
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare condition-level behavior across concat_b2 / CNNC / CCDG / WhiteningNet"
    )
    parser.add_argument("--concat-b2", type=str, required=True,
                        help="内部 concat_b2 的 run_001 目录")
    parser.add_argument("--cnnc", type=str, required=True,
                        help="外部 CNNC 实验根目录")
    parser.add_argument("--ccdg", type=str, required=True,
                        help="外部 CCDG 实验根目录")
    parser.add_argument("--whitening", type=str, required=True,
                        help="外部 WhiteningNet 实验根目录")
    parser.add_argument("--outdir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--key-conditions", type=str, nargs="*",
                        default=["S1725_L400_T6", "S1948_L400_T11"],
                        help="关键工况列表，默认 S1725 和 S1948")
    return parser.parse_args()


def main():
    args = parse_args()

    concat_b2_dir = Path(args.concat_b2)
    cnnc_dir = Path(args.cnnc)
    ccdg_dir = Path(args.ccdg)
    whitening_dir = Path(args.whitening)
    outdir = Path(args.outdir)
    key_conditions = args.key_conditions

    all_rows = []

    # 内部基线
    all_rows.extend(collect_internal_concat_b2(concat_b2_dir, method_name="concat_b2"))

    # 外部三组
    all_rows.extend(collect_external_method(cnnc_dir, method_name="CNNC"))
    all_rows.extend(collect_external_method(ccdg_dir, method_name="CCDG"))
    all_rows.extend(collect_external_method(whitening_dir, method_name="WhiteningNet"))

    df_all = build_dataframe(all_rows)

    save_outputs(df_all, outdir, key_conditions)
    print_terminal_summary(df_all, key_conditions)

    print("\n输出完成：")
    print(f"  - {outdir / 'method_condition_behavior_summary.csv'}")
    print(f"  - {outdir / 'key_condition_compare.csv'}")
    print(f"  - {outdir / 'key_condition_compare.txt'}")


if __name__ == "__main__":
    main()


"""
python compare_condition_behavior.py \
  --concat-b2 /root/py/multidiag_remote/multidiag/outputs/2026-03-10/exp_align_supcon_proto_vce_concat_b2/run_001 \
  --cnnc /root/py/multidiag_remote/DGFDBenchmark_uored/Output/CNNC_UORED/cnnc_uored_b001_allsrc_alltgt \
  --ccdg /root/py/multidiag_remote/DGFDBenchmark_uored/Output/CCDG_UORED/ccdg_uored_b001_allsrc_alltgt \
  --whitening /root/py/multidiag_remote/DGFDBenchmark_uored/Output/WhiteningNet_UORED/whiteningnet_uored_b001_allsrc_alltgt \
  --outdir /root/py/multidiag_remote/analysis_outputs/compare_condition_behavior
  
  
/root/py/multidiag_remote/analysis_outputs/compare_condition_behavior/key_condition_compare.csv

"""