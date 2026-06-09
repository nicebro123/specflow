"""Pool-based PyTorch dataset for unpaired perturbation data."""

from typing import Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


class PerturbationDataset(Dataset):
    """Randomly pair control cells with cells from a perturbation pool.

    Single-cell perturbation measurements are population-level observations,
    not paired before/after samples. Each access draws independently from the
    common control pool and the selected perturbation-condition pool.
    """

    def __init__(
        self,
        control_expression: np.ndarray,
        perturbed_by_condition: Mapping[str, np.ndarray],
        perturbation_map: Mapping[str, np.ndarray],
        conditions: Optional[Sequence[str]] = None,
        samples_per_condition: Optional[int] = None,
    ) -> None:
        self.control_expression = np.asarray(control_expression, dtype=np.float32)
        if self.control_expression.ndim != 2 or self.control_expression.shape[0] == 0:
            raise ValueError("control_expression must be a non-empty (N, G) array")

        self.n_genes = self.control_expression.shape[1]
        self.perturbed_by_condition = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in perturbed_by_condition.items()
        }
        self.conditions = list(conditions or self.perturbed_by_condition.keys())
        if not self.conditions:
            raise ValueError("at least one perturbation condition is required")
        self.perturbation_map = perturbation_map

        for name in self.conditions:
            if name not in self.perturbed_by_condition or name not in perturbation_map:
                raise KeyError(f"missing samples or perturbation mask for {name!r}")
            cells = self.perturbed_by_condition[name]
            if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] != self.n_genes:
                raise ValueError(f"invalid expression pool for {name!r}")
            if np.asarray(perturbation_map[name]).shape != (self.n_genes,):
                raise ValueError(f"perturbation mask for {name!r} must have shape (G,)")

        if samples_per_condition is None:
            samples_per_condition = max(
                self.perturbed_by_condition[name].shape[0] for name in self.conditions
            )
        self.samples_per_condition = int(samples_per_condition)
        if self.samples_per_condition < 1:
            raise ValueError("samples_per_condition must be positive")

    def __len__(self) -> int:
        return len(self.conditions) * self.samples_per_condition

    def __getitem__(self, index: int):
        condition = self.conditions[index % len(self.conditions)]
        control_idx = np.random.randint(self.control_expression.shape[0])
        perturbed = self.perturbed_by_condition[condition]
        target_idx = np.random.randint(perturbed.shape[0])
        return {
            "ctrl_expr": torch.from_numpy(self.control_expression[control_idx]),
            "pert_expr": torch.from_numpy(perturbed[target_idx]),
            "pert_mask": torch.from_numpy(
                np.asarray(self.perturbation_map[condition], dtype=np.float32)
            ),
            "condition": condition,
        }


class ConditionBatchSampler(BatchSampler):
    """Yield batches whose random target cells share one perturbation condition."""

    def __init__(
        self,
        dataset: PerturbationDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batches = []
        n_conditions = len(self.dataset.conditions)
        for condition_idx in range(n_conditions):
            indices = list(
                range(condition_idx, len(self.dataset), n_conditions)
            )
            if self.shuffle:
                np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            np.random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        per_condition = self.dataset.samples_per_condition // self.batch_size
        if not self.drop_last and self.dataset.samples_per_condition % self.batch_size:
            per_condition += 1
        return len(self.dataset.conditions) * per_condition


def make_dataloader(
    dataset: PerturbationDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    group_by_condition: bool = False,
) -> DataLoader:
    if group_by_condition:
        return DataLoader(
            dataset,
            batch_sampler=ConditionBatchSampler(dataset, batch_size, shuffle=shuffle),
            num_workers=num_workers,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
