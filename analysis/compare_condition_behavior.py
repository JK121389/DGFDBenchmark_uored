#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_condition_behavior_v2.py

用途：
统一比较四组实验（内部 concat_b2 + 外部 CNNC / CCDG / WhiteningNet）
在各 test condition 上的 condition-level 行为差异。

相比旧版优化：
1. 增加更明确的路径与文件存在性检查
2. 修正 “all_wrong_single_channel” 口径过宽的问题
3. 区分：
   - is_single_channel_dominant：错误是否高度集中于单一通道
   - is_extreme_failure：是否属于极端 failure exemplar
4. 输出更适合论文整理的 pivot 表
5. 不再依赖 to_markdown / tabulate

支持两类输入：
A. 内部实验（如 concat_b2）
   - 单个总 test_preds.csv，按 condition_id 自动拆分
B. 外部实验（如 CNNC / CCDG / WhiteningNet）
   - artifacts/test_preds__<condition>.csv
   - 可选 analysis_condition__<condition>/condition_summary.json

输出：
- method_condition_behavior_summary.csv
- key_condition_compare.csv
- key_condition_compare.txt
- method_condition_pivot_acc.csv
- method_condition_pivot_error_path.csv

示例：
python analysis/compare_condition_behavior_v2.py \
  --concat-b2 /root/py/multidiag_remote/multidiag/outputs/2026-03-10/exp_align_supcon_proto_vce_concat_b2/run_001 \
  --cnnc /root/py/multidiag_remote/DGFDBenchmark_uored/Output/CNNC_UORED/cnnc_uored_b001_allsrc_alltgt \
  --ccdg /root/py/multidiag_remote/DGFDBenchmark_uored/Output/CCDG_UORED/ccdg_uored_b001_allsrc_alltgt \
  --whitening /root/py/multidiag_remote/DGFDBenchmark_uored/Output/WhiteningNet_UORED/whiteningnet_uored_b001_allsrc_alltgt \
  --outdir /root/py/multidiag_remote/analysis_outputs/compare_condition_behavior_v2

说明：
- 默认关键工况：S1725_L400_T6, S1948_L400_T11
- 默认“极端 failure”判据：
    acc <= 0.05 且 major_error_ratio_among_wrong >= 0.95
- 默认“单通道主导”判据：
    major_error_ratio_among_wrong >= 0.95 且 n_wrong >= 20
"""


import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

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


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_label_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def extract_condition_from_filename(filename: str) -> Optional[str]:
    """
    从 test_preds__S1709_L0_T29.csv 提取 condition_id
    """
    m = re.match(r"test_preds__(.+)\.csv$", filename)
    if m:
        return m.group(1)
    return None


def check_dir_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[{name}] 目录不存在：{path}")
    if not path.is_dir():
        raise NotADirectoryError(f"[{name}] 不是目录：{path}")


def infer_internal_test_preds_path(run_dir: Path) -> Path:
    """
    内部 concat_b2 的 test_preds.csv 可能在：
    - artifacts/test_preds.csv
    - predictions/test_preds.csv
    优先 artifacts
    """
    candidates = [
        run_dir / "artifacts" / "test_preds.csv",
        run_dir / "predictions" / "test_preds.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    msg = "[concat_b2] 未找到内部实验 test_preds.csv，已检查：\n"
    msg += "\n".join([f"  - {str(p)}" for p in candidates])
    raise FileNotFoundError(msg)


def parse_external_condition_summary(summary_json: Optional[dict]) -> Dict[str, object]:
    """
    宽松提取外部 summary 中可能有帮助的字段。
    """
    if not summary_json:
        return {}

    out = {}
    for key in [
        "condition_id",
        "acc",
        "accuracy",
        "n_samples",
        "n_correct",
        "n_wrong",
        "major_error_path",
        "major_error_count",
        "major_error_ratio_among_wrong",
    ]:
        if key in summary_json:
            out[f"summary_{key}"] = summary_json[key]
    return out


# =========================
# 行为统计核心
# =========================

def compute_behavior_from_df(
    df: pd.DataFrame,
    method_name: str,
    condition_id: str,
    dominant_ratio_thr: float = 0.95,
    dominant_min_wrong: int = 20,
    extreme_acc_thr: float = 0.05,
    extreme_ratio_thr: float = 0.95,
) -> Dict[str, object]:
    """
    从逐样本预测表中，统计 condition 级行为摘要。
    """

    true_col = find_first_existing(df, TRUE_COL_CANDIDATES)
    pred_col = find_first_existing(df, PRED_COL_CANDIDATES)

    if true_col is None:
        raise ValueError(
            f"[{method_name} | {condition_id}] 未找到真实标签列。\n"
            f"候选：{TRUE_COL_CANDIDATES}\n"
            f"当前列：{list(df.columns)}"
        )
    if pred_col is None:
        raise ValueError(
            f"[{method_name} | {condition_id}] 未找到预测标签列。\n"
            f"候选：{PRED_COL_CANDIDATES}\n"
            f"当前列：{list(df.columns)}"
        )

    y_true = normalize_label_series(df[true_col])
    y_pred = normalize_label_series(df[pred_col])

    n_samples = int(len(df))
    correct_mask = (y_true == y_pred)
    n_correct = int(correct_mask.sum())
    n_wrong = int((~correct_mask).sum())
    acc = float(n_correct / n_samples) if n_samples > 0 else 0.0

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

    # 更严格地区分“单通道主导”与“极端 failure”
    is_single_channel_dominant = bool(
        n_wrong >= dominant_min_wrong and
        major_error_ratio_among_wrong >= dominant_ratio_thr
    )

    is_extreme_failure = bool(
        acc <= extreme_acc_thr and
        n_wrong > 0 and
        major_error_ratio_among_wrong >= extreme_ratio_thr
    )

    # 给出一个便于论文描述的 failure 类型标签
    if n_wrong == 0:
        failure_mode = "clean"
    elif is_extreme_failure:
        failure_mode = "extreme_single_channel_failure"
    elif is_single_channel_dominant:
        failure_mode = "single_channel_dominant_error"
    else:
        failure_mode = "mixed_error"

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
        "is_single_channel_dominant": is_single_channel_dominant,
        "is_extreme_failure": is_extreme_failure,
        "failure_mode": failure_mode,
        "true_col_used": true_col,
        "pred_col_used": pred_col,
    }


# =========================
# 两类实验读取逻辑
# =========================

def collect_internal_concat_b2(
    run_dir: Path,
    method_name: str = "concat_b2",
    dominant_ratio_thr: float = 0.95,
    dominant_min_wrong: int = 20,
    extreme_acc_thr: float = 0.05,
    extreme_ratio_thr: float = 0.95,
) -> List[Dict[str, object]]:
    """
    内部实验：单个总 test_preds.csv，需要按 condition_id 分组统计。
    """
    check_dir_exists(run_dir, method_name)
    test_preds_path = infer_internal_test_preds_path(run_dir)
    df = safe_read_csv(test_preds_path)

    cond_col = find_first_existing(df, COND_COL_CANDIDATES)
    if cond_col is None:
        raise ValueError(
            f"[{method_name}] 未找到 condition_id 列。\n"
            f"候选：{COND_COL_CANDIDATES}\n"
            f"当前列：{list(df.columns)}"
        )

    rows = []
    for condition_id, sub_df in df.groupby(cond_col):
        condition_id = str(condition_id)
        row = compute_behavior_from_df(
            sub_df,
            method_name=method_name,
            condition_id=condition_id,
            dominant_ratio_thr=dominant_ratio_thr,
            dominant_min_wrong=dominant_min_wrong,
            extreme_acc_thr=extreme_acc_thr,
            extreme_ratio_thr=extreme_ratio_thr,
        )
        row["source_type"] = "internal_single_csv"
        row["source_csv"] = str(test_preds_path)
        row["cond_col_used"] = cond_col
        rows.append(row)

    rows.sort(key=lambda x: x["condition_id"])
    return rows


def collect_external_method(
    exp_dir: Path,
    method_name: str,
    dominant_ratio_thr: float = 0.95,
    dominant_min_wrong: int = 20,
    extreme_acc_thr: float = 0.05,
    extreme_ratio_thr: float = 0.95,
) -> List[Dict[str, object]]:
    """
    外部实验：artifacts/test_preds__<condition>.csv
    """
    check_dir_exists(exp_dir, method_name)

    artifacts_dir = exp_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"[{method_name}] 未找到 artifacts 目录：{artifacts_dir}")

    csv_files = sorted(artifacts_dir.glob("test_preds__*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"[{method_name}] 未找到 test_preds__*.csv：{artifacts_dir}")

    rows = []
    for csv_path in csv_files:
        condition_id = extract_condition_from_filename(csv_path.name)
        if condition_id is None:
            continue

        df = safe_read_csv(csv_path)
        row = compute_behavior_from_df(
            df,
            method_name=method_name,
            condition_id=condition_id,
            dominant_ratio_thr=dominant_ratio_thr,
            dominant_min_wrong=dominant_min_wrong,
            extreme_acc_thr=extreme_acc_thr,
            extreme_ratio_thr=extreme_ratio_thr,
        )

        summary_json_path = exp_dir / f"analysis_condition__{condition_id}" / "condition_summary.json"
        summary_json = load_json_if_exists(summary_json_path)
        row.update(parse_external_condition_summary(summary_json))

        row["source_type"] = "external_per_condition_csv"
        row["source_csv"] = str(csv_path)
        row["summary_json_path"] = str(summary_json_path) if summary_json_path.exists() else ""
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
        "is_single_channel_dominant",
        "is_extreme_failure",
        "failure_mode",
        "source_type",
        "source_csv",
        "summary_json_path",
        "cond_col_used",
        "true_col_used",
        "pred_col_used",
        # 可选 summary 提取字段
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

    df_all_sorted = df_all.sort_values(["condition_id", "method"]).reset_index(drop=True)
    df_key = df_all_sorted[df_all_sorted["condition_id"].isin(key_conditions)].copy()

    # 1. 全表
    all_csv = outdir / "method_condition_behavior_summary.csv"
    df_all_sorted.to_csv(all_csv, index=False, encoding="utf-8-sig")

    # 2. 关键工况表
    key_csv = outdir / "key_condition_compare.csv"
    df_key.to_csv(key_csv, index=False, encoding="utf-8-sig")

    # 3. 文本预览
    key_txt = outdir / "key_condition_compare.txt"
    with open(key_txt, "w", encoding="utf-8") as f:
        f.write("Key Condition Compare\n")
        f.write("=" * 120 + "\n\n")
        if len(df_key) == 0:
            f.write("No matched key conditions found.\n")
        else:
            preview_cols = [
                "method", "condition_id", "n_samples", "acc",
                "major_error_path", "major_error_count",
                "major_error_ratio_among_wrong",
                "is_single_channel_dominant",
                "is_extreme_failure",
                "failure_mode",
            ]
            preview_cols = [c for c in preview_cols if c in df_key.columns]
            f.write(df_key[preview_cols].to_string(index=False))
            f.write("\n")

    # 4. acc pivot
    acc_pivot = df_all_sorted.pivot(index="condition_id", columns="method", values="acc")
    acc_pivot = acc_pivot.sort_index()
    acc_pivot.to_csv(outdir / "method_condition_pivot_acc.csv", encoding="utf-8-sig")

    # 5. 主错通道 pivot
    err_pivot = df_all_sorted.pivot(index="condition_id", columns="method", values="major_error_path")
    err_pivot = err_pivot.sort_index()
    err_pivot.to_csv(outdir / "method_condition_pivot_error_path.csv", encoding="utf-8-sig")

    # 6. failure mode pivot
    mode_pivot = df_all_sorted.pivot(index="condition_id", columns="method", values="failure_mode")
    mode_pivot = mode_pivot.sort_index()
    mode_pivot.to_csv(outdir / "method_condition_pivot_failure_mode.csv", encoding="utf-8-sig")


def print_terminal_summary(df_all: pd.DataFrame, key_conditions: List[str]) -> None:
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 220)

    show_cols = [
        "method",
        "condition_id",
        "n_samples",
        "acc",
        "major_error_path",
        "major_error_count",
        "major_error_ratio_among_wrong",
        "is_single_channel_dominant",
        "is_extreme_failure",
        "failure_mode",
    ]
    show_cols = [c for c in show_cols if c in df_all.columns]

    print("\n================ 全部 condition 行为摘要 ================\n")
    print(
        df_all.sort_values(["condition_id", "method"])[show_cols].to_string(index=False)
    )

    df_key = df_all[df_all["condition_id"].isin(key_conditions)].copy()
    print("\n================ 关键工况对照 ================\n")
    if len(df_key) == 0:
        print("未匹配到关键工况。")
    else:
        print(
            df_key.sort_values(["condition_id", "method"])[show_cols].to_string(index=False)
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
                        help="关键工况列表")

    parser.add_argument("--dominant-ratio-thr", type=float, default=0.95,
                        help="判定单通道主导错误的错误集中度阈值")
    parser.add_argument("--dominant-min-wrong", type=int, default=20,
                        help="判定单通道主导错误的最小错误数阈值")
    parser.add_argument("--extreme-acc-thr", type=float, default=0.05,
                        help="判定极端 failure 的 acc 上限阈值")
    parser.add_argument("--extreme-ratio-thr", type=float, default=0.95,
                        help="判定极端 failure 的错误集中度阈值")

    return parser.parse_args()


def main():
    args = parse_args()

    concat_b2_dir = Path(args.concat_b2)
    cnnc_dir = Path(args.cnnc)
    ccdg_dir = Path(args.ccdg)
    whitening_dir = Path(args.whitening)
    outdir = Path(args.outdir)

    key_conditions = args.key_conditions
    dominant_ratio_thr = args.dominant_ratio_thr
    dominant_min_wrong = args.dominant_min_wrong
    extreme_acc_thr = args.extreme_acc_thr
    extreme_ratio_thr = args.extreme_ratio_thr

    all_rows: List[Dict[str, object]] = []

    # 内部基线
    all_rows.extend(
        collect_internal_concat_b2(
            concat_b2_dir,
            method_name="concat_b2",
            dominant_ratio_thr=dominant_ratio_thr,
            dominant_min_wrong=dominant_min_wrong,
            extreme_acc_thr=extreme_acc_thr,
            extreme_ratio_thr=extreme_ratio_thr,
        )
    )

    # 外部三组
    all_rows.extend(
        collect_external_method(
            cnnc_dir,
            method_name="CNNC",
            dominant_ratio_thr=dominant_ratio_thr,
            dominant_min_wrong=dominant_min_wrong,
            extreme_acc_thr=extreme_acc_thr,
            extreme_ratio_thr=extreme_ratio_thr,
        )
    )

    all_rows.extend(
        collect_external_method(
            ccdg_dir,
            method_name="CCDG",
            dominant_ratio_thr=dominant_ratio_thr,
            dominant_min_wrong=dominant_min_wrong,
            extreme_acc_thr=extreme_acc_thr,
            extreme_ratio_thr=extreme_ratio_thr,
        )
    )

    all_rows.extend(
        collect_external_method(
            whitening_dir,
            method_name="WhiteningNet",
            dominant_ratio_thr=dominant_ratio_thr,
            dominant_min_wrong=dominant_min_wrong,
            extreme_acc_thr=extreme_acc_thr,
            extreme_ratio_thr=extreme_ratio_thr,
        )
    )

    df_all = build_dataframe(all_rows)
    save_outputs(df_all, outdir, key_conditions)
    print_terminal_summary(df_all, key_conditions)

    print("\n输出完成：")
    print(f"  - {outdir / 'method_condition_behavior_summary.csv'}")
    print(f"  - {outdir / 'key_condition_compare.csv'}")
    print(f"  - {outdir / 'key_condition_compare.txt'}")
    print(f"  - {outdir / 'method_condition_pivot_acc.csv'}")
    print(f"  - {outdir / 'method_condition_pivot_error_path.csv'}")
    print(f"  - {outdir / 'method_condition_pivot_failure_mode.csv'}")


if __name__ == "__main__":
    main()

"""
python analysis/compare_condition_behavior.py \
  --concat-b2 /root/py/multidiag_remote/multidiag/outputs/2026-03-10/exp_align_supcon_proto_vce_concat_b2/run_001 \
  --cnnc /root/py/multidiag_remote/DGFDBenchmark_uored/Output/CNNC_UORED/cnnc_uored_b001_allsrc_alltgt \
  --ccdg /root/py/multidiag_remote/DGFDBenchmark_uored/Output/CCDG_UORED/ccdg_uored_b001_allsrc_alltgt \
  --whitening /root/py/multidiag_remote/DGFDBenchmark_uored/Output/WhiteningNet_UORED/whiteningnet_uored_b001_allsrc_alltgt \
  --outdir /root/py/multidiag_remote/analysis_outputs/compare_condition_behavior


"""