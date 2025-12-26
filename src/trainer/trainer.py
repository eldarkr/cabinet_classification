from pathlib import Path
from omegaconf import DictConfig
import torch
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.data_utils import get_class_balances
from src.utils.logger import get_logger

logger = get_logger(__name__)


# TODO: use factory pattern to build different models/optimizers/losses for better experiments management
class Trainer:
    def __init__(self, config: DictConfig):
        self.config = config

        # device сonfiguration
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # apple silicon
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        # model
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, config.model.num_classes)
        self.model = self.model.to(self.device)

        # model save path (always relative to project root)
        project_root = Path(__file__).resolve().parents[2]
        self.model_dir = (project_root / config.model.save_path).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / config.model.experiment_name

        # optimizer
        self.learning_rate = config.optimizer.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.training.num_epochs, 
            eta_min=1e-6
        )
        
        # loss function
        class_weights = None
        if config.model.use_class_weights:
            class_weights = get_class_balances(config).to(self.device)
            logger.info(f"Using class weights: {class_weights}")
        self.loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

        # training config
        self.num_epochs = config.training.num_epochs
        self.early_stopping_patience = config.training.early_stopping.get("patience", None)
        self.early_stopping_min_delta = config.training.early_stopping.get("min_delta", 0.0)

    def configure_finetune(self):
        if self.config.training.fine_tune.strategy == "frozen":
            logger.info("Freezing all layers except the final layer for fine-tuning.")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            logger.info("Fine-tuning all layers.")

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0

        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_function(logits, y)
            loss.backward()
            # for sampler stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        
        self.scheduler.step()

        return total_loss / len(data_loader)

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct, total = 0, 0

        all_preds = []
        all_targets = []

        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.loss_function(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        metrics = {
            "accuracy": accuracy,
            "preds":  torch.cat(all_preds),
            "targets": torch.cat(all_targets)
        }
        return avg_loss, metrics

    def fit(self, train_loader, val_loader):
        history = {"train_loss": [], "val_loss": [], "accuracy": [], "preds": [], "targets": []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
    
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["accuracy"].append(val_metrics["accuracy"])
            history["preds"].append(val_metrics["preds"])
            history["targets"].append(val_metrics["targets"])
            
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}"
            )

            if self.early_stopping_patience is None:
                continue
            
            if val_loss < best_val_loss - self.early_stopping_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                self.save_model()
                continue
            
            patience_counter += 1
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break
        return history
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self, model_path=None):
        """Load model weights from file."""
        if model_path is None:
            model_path = self.model_path
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from: {model_path}")
