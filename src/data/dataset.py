from pathlib import Path
from typing import Optional, Callable, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class CabinetDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[Callable] = None, class_transforms: Optional[dict] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_transforms = class_transforms if class_transforms is not None else {}

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._load_data()

    def _load_data(self):
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        for class_name in classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = self.data_dir / class_name

            for img_path in class_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.labels[idx]
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        class_name = self.idx_to_class[label]
        if class_name in self.class_transforms:
            class_transform = self.class_transforms[class_name]
            image = class_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.image_paths)
