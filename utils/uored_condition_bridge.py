from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class UoredItem:
    sample_id: str
    file_id: str
    raw_path: str
    mat_key: str
    label: int
    condition_id: str
    fs: int
    start: int
    length: int
    vib_col: int
    aud_col: int


class UoredVafclsDataset(Dataset):
    """Minimal self-contained UORED dataset for bridge use.

    split is selected from file_id groups in split json.
    Returns a dict with x_v / x_a / y / meta.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split_path: Union[str, Path],
        split: str,
        root: Optional[Union[str, Path]] = None,
        cache_mat: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split_path = Path(split_path)
        self.split = split
        self.root = Path(root) if root is not None else Path('.')
        self.cache_mat = cache_mat
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be one of train/val/test, got: {split}")
        self._split_files = self._load_split_files(self.split_path, split)
        self.items: List[UoredItem] = self._load_manifest(self.manifest_path, self._split_files)
        self._mat_cache: Dict[str, np.ndarray] = {}
        if not self.items:
            raise RuntimeError(f"No samples found for split='{split}'.")

    @staticmethod
    def _load_split_files(split_path: Path, split: str) -> set:
        with open(split_path, 'r', encoding='utf-8') as f:
            j = json.load(f)
        key = f'{split}_files'
        if key not in j:
            raise KeyError(f'Split json missing key: {key}')
        return set(j[key])

    @staticmethod
    def _load_manifest(manifest_path: Path, allowed_files: set) -> List[UoredItem]:
        items: List[UoredItem] = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            required = {
                'sample_id', 'file_id', 'raw_path', 'mat_key', 'label',
                'condition_id', 'fs', 'start', 'length', 'vib_col', 'aud_col'
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise KeyError(f'Manifest missing columns: {sorted(list(missing))}')
            for r in reader:
                file_id = r['file_id']
                if file_id not in allowed_files:
                    continue
                items.append(UoredItem(
                    sample_id=r['sample_id'],
                    file_id=file_id,
                    raw_path=r['raw_path'],
                    mat_key=r['mat_key'],
                    label=int(r['label']),
                    condition_id=r.get('condition_id', ''),
                    fs=int(r['fs']),
                    start=int(r['start']),
                    length=int(r['length']),
                    vib_col=int(r['vib_col']),
                    aud_col=int(r['aud_col']),
                ))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _load_mat_array(self, abs_mat_path: Path, mat_key: str) -> np.ndarray:
        k = str(abs_mat_path)
        if self.cache_mat and k in self._mat_cache:
            return self._mat_cache[k]
        mat = sio.loadmat(abs_mat_path)
        if mat_key not in mat:
            keys = [kk for kk in mat.keys() if not kk.startswith('__')]
            raise KeyError(f"MAT key '{mat_key}' not found in {abs_mat_path.name}. keys={keys}")
        data = mat[mat_key]
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError(f'Unexpected MAT data shape in {abs_mat_path.name}: {getattr(data, "shape", None)}')
        if self.cache_mat:
            self._mat_cache[k] = data
        return data

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]
        abs_mat_path = (self.root / Path(it.raw_path.replace('\\', '/'))).resolve()
        data = self._load_mat_array(abs_mat_path, it.mat_key)
        s = it.start
        e = s + it.length
        vib = data[s:e, it.vib_col].astype(np.float32, copy=False)
        aud = data[s:e, it.aud_col].astype(np.float32, copy=False)
        return {
            'x_v': torch.from_numpy(vib),
            'x_a': torch.from_numpy(aud),
            'y': torch.tensor(it.label, dtype=torch.long),
            'meta': {
                'sample_id': it.sample_id,
                'file_id': it.file_id,
                'condition_id': it.condition_id,
                'fs': it.fs,
                'start': it.start,
                'length': it.length,
                'raw_path': it.raw_path,
                'mat_key': it.mat_key,
            }
        }


class _TrainTupleDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int], modality: str = 'vib'):
        self.base = base
        self.indices = list(indices)
        self.modality = modality

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        out = self.base[self.indices[idx]]
        x = out['x_v'] if self.modality == 'vib' else out['x_a']
        return x.unsqueeze(0), out['y']


class _TestTupleDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int], modality: str = 'vib'):
        self.base = base
        self.indices = list(indices)
        self.modality = modality

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        out = self.base[self.indices[idx]]
        x = out['x_v'] if self.modality == 'vib' else out['x_a']
        return x.unsqueeze(0), out['y'], out['meta']


class InfiniteLoader:
    """Simple infinite iterator wrapper over a finite DataLoader."""

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = False, drop_last: bool = True):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self._iterator = iter(self.loader)

    def __iter__(self):
        while True:
            try:
                yield next(self._iterator)
            except StopIteration:
                self._iterator = iter(self.loader)
                yield next(self._iterator)


def _condition_to_indices(base_dataset: UoredVafclsDataset) -> Dict[str, List[int]]:
    cond2idx: Dict[str, List[int]] = {}
    for i, item in enumerate(base_dataset.items):
        cond2idx.setdefault(item.condition_id, []).append(i)
    return cond2idx


def _normalize_condition_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return [x.strip() for x in value.split(',') if x.strip()]
    return [str(x) for x in value]


def build_uored_condition_loaders(configs):
    """Build CNNC-compatible loaders for UORED condition-as-domain.

    Returns:
        train_loaders_src, test_loaders_tgt, test_loaders_src, target_names, source_names
    """
    modality = getattr(configs, 'modality', 'vib')
    batch_size = int(getattr(configs, 'batch_size', 64))
    num_workers = int(getattr(configs, 'num_workers', 0))
    pin_memory = bool(getattr(configs, 'pin_memory', False))

    train_base = UoredVafclsDataset(configs.manifest_path, configs.split_path, 'train', root=configs.data_root)
    test_base = UoredVafclsDataset(configs.manifest_path, configs.split_path, 'test', root=configs.data_root)

    train_map = _condition_to_indices(train_base)
    test_map = _condition_to_indices(test_base)

    source_conditions = _normalize_condition_list(getattr(configs, 'source_conditions', None))
    target_conditions = _normalize_condition_list(getattr(configs, 'target_conditions', None))
    if source_conditions is None:
        source_conditions = sorted(train_map.keys())
    if target_conditions is None:
        target_conditions = sorted(test_map.keys())

    missing_src = [c for c in source_conditions if c not in train_map]
    missing_tgt = [c for c in target_conditions if c not in test_map]
    if missing_src:
        raise KeyError(f'Source conditions not found in train split: {missing_src}')
    if missing_tgt:
        raise KeyError(f'Target conditions not found in test split: {missing_tgt}')

    overlap = sorted(set(source_conditions) & set(target_conditions))
    if overlap and bool(getattr(configs, 'enforce_disjoint_source_target', True)):
        raise ValueError(f'Source/target conditions overlap: {overlap}')

    train_loaders_src = []
    test_loaders_src = []
    for cond in source_conditions:
        tr_ds = _TrainTupleDataset(train_base, train_map[cond], modality=modality)
        te_ds = _TestTupleDataset(train_base, train_map[cond], modality=modality)
        train_loaders_src.append(
            iter(InfiniteLoader(
                tr_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            ))
        )
        test_loaders_src.append(
            DataLoader(
                te_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
        )

    test_loaders_tgt = []
    for cond in target_conditions:
        te_ds = _TestTupleDataset(test_base, test_map[cond], modality=modality)
        test_loaders_tgt.append(
            DataLoader(
                te_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
        )

    return train_loaders_src, test_loaders_tgt, test_loaders_src, target_conditions, source_conditions