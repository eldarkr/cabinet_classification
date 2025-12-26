from pathlib import Path
from omegaconf import DictConfig
import torch
from torchvision import models

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Inference:
    def __init__(self, cfg: DictConfig):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # apple silicon
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / cfg.model.save_path / cfg.model.experiment_name
        logger.info(f"Loading model from: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        # rebuild model (will be in factory pattern later)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, cfg.model.num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        logits = self.model(x)
        preds = logits.argmax(dim=1)
        return preds.cpu()
