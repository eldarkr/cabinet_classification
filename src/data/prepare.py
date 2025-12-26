import json
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np
from omegaconf import DictConfig

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineComponent(ABC):
    def __init__(self, config: DictConfig):
        self.cfg = config

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass


class ProjectScanner(PipelineComponent):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.raw_data_dir = Path(self.cfg.data.raw_data_dir)

    def validate(self) -> None:
        """Validate that raw data directory exists."""
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_data_dir}"
            )
        logger.info(f"Raw data directory validated: {self.raw_data_dir}")

    def execute(self) -> List[Path]:
        """Scan and return list of project directories.

        Returns:
            List of project directory paths
        """
        self.validate()

        projects = []
        for item in self.raw_data_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # Check if has page folders
                page_folders = [
                    f for f in item.iterdir() if f.is_dir() and "_" in f.name
                ]
                if page_folders:
                    projects.append(item)

        projects = sorted(projects)
        logger.info(f"Found {len(projects)} projects")
        return projects

    def get_page_folders(self, project_dir: Path) -> List[Path]:
        """Find page folders within project directory.

        Args:
            project_dir: Project directory path

        Returns:
            List of page folder paths (sorted by page number)
        """
        project_name = project_dir.name
        page_folders = []

        for folder in project_dir.iterdir():
            if folder.is_dir() and folder.name.startswith(project_name + "_"):
                parts = folder.name.split("_")
                if parts[-1].isdigit():
                    page_folders.append(folder)

        return sorted(page_folders)


class ImageLoader(PipelineComponent):
    def execute(self, page_folder: Path) -> Optional[np.ndarray]:
        """Load page image as grayscale.

        Args:
            page_folder: Path to page folder containing PNG image

        Returns:
            Grayscale image array (H, W) uint8, or None if failed
        """
        page_name = page_folder.name
        image_path = page_folder / f"{page_name}.png"

        if not image_path.exists():
            logger.debug(f"Image not found: {image_path}")
            return None

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
        return img


class AnnotationParser(PipelineComponent):
    """Parses annotation JSON files."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.target_categories = set(self.cfg.data.target_categories)

    def execute(self, page_folder: Path) -> List[Dict]:
        """Parse annotations from page's *_simple.json file.

        Args:
            page_folder: Path to page folder

        Returns:
            List of annotation dicts for target categories
        """
        page_name = page_folder.name
        json_path = page_folder / f"{page_name}_simple.json"

        if not json_path.exists():
            logger.debug(f"Annotation file not found: {json_path}")
            return []

        try:
            with open(json_path) as f:
                data = json.load(f)

            # Filter annotations for target categories
            target_annotations = []
            for ann in data.get("annotations", []):
                if ann.get("label") in self.target_categories:
                    target_annotations.append(ann)

            return target_annotations
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {json_path}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error parsing annotations {json_path}: {e}")
            return []


class CropExtractor(PipelineComponent):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.padding = self.cfg.data.crop_padding

    def execute(
        self, image: np.ndarray, coords: Dict[str, int]
    ) -> Optional[np.ndarray]:
        """Extract crop from image using object_coordinates with padding.

        Args:
            image: Source image array
            coords: object_coordinates dict with x0, y0, x1, y1

        Returns:
            Cropped image array or None if invalid
        """
        try:
            x0, y0 = coords["x0"], coords["y0"]
            x1, y1 = coords["x1"], coords["y1"]
        except KeyError as e:
            logger.warning(f"Missing coordinate key: {e}")
            return None

        # Apply padding
        x0 = max(0, x0 - self.padding)
        y0 = max(0, y0 - self.padding)
        x1 = min(image.shape[1], x1 + self.padding)
        y1 = min(image.shape[0], y1 + self.padding)

        # Validate crop dimensions
        if x1 <= x0 or y1 <= y0 or (x1 - x0) < 5 or (y1 - y0) < 5:
            logger.debug(f"Invalid crop dimensions: ({x0}, {y0}) to ({x1}, {y1})")
            return None

        return image[y0:y1, x0:x1]


class CropSaver(PipelineComponent):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.crops_dir = Path(self.cfg.data.crops_dir)
        self.target_categories = set(self.cfg.data.target_categories)
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create output directories for each category."""
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        for label in self.target_categories:
            (self.crops_dir / label).mkdir(exist_ok=True)
        logger.info(f"Output directories created: {self.crops_dir}")

    def execute(self, crop: np.ndarray, label: str, filename: str) -> bool:
        """Save crop to appropriate directory.

        Args:
            crop: Image crop array
            label: Category label
            filename: Filename for the crop

        Returns:
            True if saved successfully, False otherwise
        """
        crop_path = self.crops_dir / label / filename

        try:
            success = cv2.imwrite(str(crop_path), crop)
            if not success:
                logger.warning(f"cv2.imwrite returned False for: {crop_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to save crop {crop_path}: {e}")
            return False


class DataPreparationPipeline:
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.project_scanner = ProjectScanner(config)
        self.image_loader = ImageLoader(config)
        self.annotation_parser = AnnotationParser(config)
        self.crop_extractor = CropExtractor(config)
        self.crop_saver = CropSaver(config)

    def process_page(
        self, page_folder: Path, project_name: str, label_counts: Dict[str, int]
    ) -> int:
        """Process single page: extract and save all crops.

        Args:
            page_folder: Path to page folder
            project_name: Project name
            label_counts: Dict to track crop counts per label

        Returns:
            Number of crops extracted
        """
        # Load page image
        image = self.image_loader.execute(page_folder)
        if image is None:
            return 0

        # Parse annotations
        annotations = self.annotation_parser.execute(page_folder)
        if not annotations:
            return 0

        # Extract page number from folder name
        page_num = int(page_folder.name.split("_")[-1])

        crops_saved = 0
        for ann in annotations:
            # Extract crop
            crop = self.crop_extractor.execute(image, ann["object_coordinates"])
            if crop is None:
                continue

            # Generate filename and save
            label = ann["label"]
            crop_name = f"{project_name}_p{page_num:05d}_a{ann['id']:04d}.png"

            if self.crop_saver.execute(crop, label, crop_name):
                label_counts[label] += 1
                crops_saved += 1

        return crops_saved

    def run(self) -> Dict[str, int]:
        """Execute full data preparation pipeline.

        Returns:
            Dictionary with statistics (total_crops, label_counts)
        """
        # Scan projects
        projects = self.project_scanner.execute()

        if not projects:
            logger.warning("No projects found!")
            return {"total_crops": 0, "num_projects": 0}

        # Process all pages from all projects
        label_counts = defaultdict(int)
        total_crops = 0

        for i, project_dir in enumerate(projects, 1):
            logger.info(f"Processing project {i}/{len(projects)}: {project_dir.name}")
            project_name = project_dir.name
            page_folders = self.project_scanner.get_page_folders(project_dir)

            if not page_folders:
                logger.warning(f"No page folders found in {project_name}")
                continue

            for page_folder in page_folders:
                crops_saved = self.process_page(page_folder, project_name, label_counts)
                total_crops += crops_saved

        # Log final statistics
        logger.info(f"Extracted {total_crops} crops from {len(projects)} projects")

        if total_crops == 0:
            logger.warning("No crops extracted! Check target_categories and data.")

        return {
            "total_crops": total_crops,
            "num_projects": len(projects),
            "label_counts": dict(label_counts),
        }
