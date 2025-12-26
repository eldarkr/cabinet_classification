from pathlib import Path
from PIL import Image
import torch

from src.data.dataset import CabinetDataset


def get_class_balances(config):
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / config.data.crops_dir
    dataset = CabinetDataset(
        data_dir=dataset_path,
        transform=None
    )
    targets = [label for _, label in dataset]
    class_counts = torch.bincount(torch.tensor(targets), minlength=config.model.num_classes)
    weights = 1.0 / class_counts.float()
    normalized_weights = weights / weights.sum() * config.model.num_classes
    return normalized_weights


def save_learning_history(history, save_path):
    save_path = save_path.resolve()  # for compatibility
    torch.save(history, save_path)


def get_label_name(label_idx, cfg):
    target_names = sorted(cfg.data.target_categories)
    return target_names[label_idx]


def load_images(image_paths, transform):
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        images.append(img)
    batch = torch.stack(images)
    return batch
