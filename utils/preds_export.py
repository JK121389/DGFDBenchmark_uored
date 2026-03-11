from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def collect_batch_meta(meta_batch: Any) -> List[Dict[str, Any]]:
    """Convert collated meta into a list of per-sample dicts."""
    if meta_batch is None:
        return []
    if isinstance(meta_batch, list):
        return [dict(x) for x in meta_batch]
    if isinstance(meta_batch, dict):
        keys = list(meta_batch.keys())
        if not keys:
            return []
        n = len(meta_batch[keys[0]])
        out = []
        for i in range(n):
            rec = {}
            for k in keys:
                v = meta_batch[k]
                rec[k] = v[i]
            out.append(rec)
        return out
    raise TypeError(f'Unsupported meta batch type: {type(meta_batch)}')


def write_lightweight_preds_csv(records: List[Dict[str, Any]], out_csv: str) -> None:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    required = ['sample_id', 'file_id', 'condition_id', 'y_true', 'y_pred']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required prediction columns: {missing}')
    preferred = ['sample_id', 'file_id', 'condition_id', 'y_true', 'y_pred', 'method', 'run_id', 'domain_split']
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]
    df.to_csv(path, index=False, encoding='utf-8-sig')