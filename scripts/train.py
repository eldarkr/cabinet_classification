from omegaconf import OmegaConf

from src.data.dataloader import build_dataloaders
from src.data.transforms import get_transforms
from src.trainer.trainer import Trainer
from src.utils.set_seed import set_seed
from src.utils.data_utils import save_learning_history
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    config_path = "configs/config.yaml"
    cfg = OmegaConf.load(config_path)
    cli_args = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_args)
    set_seed(cfg.random_seed)

    train_transform, val_transform, class_transforms_dict = get_transforms(cfg)

    train_loader, val_loader = build_dataloaders(cfg, train_transform, val_transform, class_transforms_dict)
    trainer = Trainer(cfg)
    trainer.configure_finetune()
    history = trainer.fit(train_loader, val_loader)
    history_save_path = trainer.model_dir / f"history_{cfg.model.experiment_name}"
    save_learning_history(history, history_save_path)
    logger.info(f"Saving learning history to: {history_save_path}")


if __name__ == "__main__":
    main()
