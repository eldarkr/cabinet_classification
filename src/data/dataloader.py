from pathlib import Path
from typing import Optional, Callable, Tuple
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.dataset import CabinetDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_sample_counts = np.bincount(labels)
    weight_per_class = 1.0 / class_sample_counts
    weights = weight_per_class[labels]
    weights = torch.as_tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


def build_dataloaders(
    cfg,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    class_transforms_dict: Optional[dict] = None
) -> Tuple[DataLoader, DataLoader]:
    project_root = Path(__file__).parent.parent.parent  # classification/src/data/dataloader.py
    data_dir = project_root / cfg.data.crops_dir
    data_dir = data_dir.resolve()  # for compatibility

    base_ds = CabinetDataset(data_dir)
    labels = np.array(base_ds.labels)

    # stratified split for disbalanced dataset
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=cfg.data.val_split,
        random_state=cfg.random_seed,
    )

    train_idx, val_idx = next(
        splitter.split(np.zeros(len(labels)), labels)
    )

    train_ds = copy.copy(base_ds)
    train_ds.transform = train_transform
    train_ds.class_transforms = class_transforms_dict or {}

    val_ds = copy.copy(base_ds)
    val_ds.transform = val_transform
    val_ds.class_transforms = class_transforms_dict or {}

    train_ds = Subset(train_ds, train_idx)
    val_ds = Subset(val_ds, val_idx)

    if cfg.data.sampler.enable:
        logger.info("Using WeightedRandomSampler for train data loader.")
        train_labels = labels[train_idx]
        sampler = setup_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler=sampler if cfg.data.sampler.enable else None
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False
    )

    return train_loader, val_loader
