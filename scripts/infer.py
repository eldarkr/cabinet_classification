from pathlib import Path
from omegaconf import OmegaConf

from src.utils.data_utils import get_label_name, load_images
from src.utils.logger import get_logger
from src.utils.set_seed import set_seed
from src.inference.inference import Inference
from src.data.transforms import get_transforms

logger = get_logger(__name__)


def main():
    cfg = OmegaConf.load("configs/config.yaml")
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    set_seed(cfg.random_seed)

    _, val_transform, _ = get_transforms(cfg)
    inferencer = Inference(cfg)

    image_path = cfg.inference.image_path
    image_path = Path(image_path)
    if image_path.is_dir():
        image_files = sorted([p for p in image_path.glob("*.png")])
        if not image_files:
            logger.warning(f"No PNG images found in {image_path}")
            return
    else:
        image_files = [image_path]

    batch = load_images(image_files, val_transform)
    preds = inferencer.predict(batch)

    for path, pred in zip(image_files, preds):
        label_name = get_label_name(pred.item(), cfg)
        logger.info(f"{path}  -->  class {label_name}")


if __name__ == "__main__":
    main()
