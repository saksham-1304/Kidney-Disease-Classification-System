import os
import sys
import shutil
import random
import subprocess
import zipfile
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig


# Name of the extracted dataset folder inside artifacts/data_ingestion/
_DATASET_DIR_NAME = "kidney-ct-scan-dataset"


def _kaggle_cli() -> str:
    """Return the absolute path to the kaggle CLI in the current venv."""
    scripts_dir = os.path.dirname(sys.executable)
    for name in ("kaggle.exe", "kaggle"):
        path = os.path.join(scripts_dir, name)
        if os.path.isfile(path):
            return path
    return "kaggle"  # fallback to PATH


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> None:
        """Download the CT Kidney dataset from Kaggle using the kaggle CLI.

        Downloads a zip archive and extracts it into
        ``<unzip_dir>/<_DATASET_DIR_NAME>/``.
        """
        dest_dir = os.path.join(
            str(self.config.unzip_dir), _DATASET_DIR_NAME
        )

        if os.path.isdir(dest_dir) and os.listdir(dest_dir):
            logger.info(
                f"Dataset already exists at {dest_dir}, skipping download."
            )
            return

        download_dir = str(self.config.unzip_dir)
        os.makedirs(download_dir, exist_ok=True)

        dataset_slug = self.config.source_URL
        logger.info(f"Downloading dataset from Kaggle: {dataset_slug}")

        # Use kaggle CLI to download the zip
        subprocess.run(
            [
                _kaggle_cli(), "datasets", "download",
                "-d", dataset_slug,
                "-p", download_dir,
            ],
            check=True,
        )

        # The zip file is named after the last component of the slug
        zip_name = dataset_slug.split("/")[-1] + ".zip"
        zip_path = os.path.join(download_dir, zip_name)
        logger.info(f"Extracting {zip_path}...")

        # Extract into a temporary directory, then locate the class folders
        tmp_extract = os.path.join(download_dir, "_extracted_tmp")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_extract)

        # The extracted content may have a top-level folder or the class
        # dirs directly.  Detect and normalise.
        contents = [
            d for d in os.listdir(tmp_extract)
            if os.path.isdir(os.path.join(tmp_extract, d))
        ]
        source = tmp_extract
        if len(contents) == 1:
            candidate = os.path.join(tmp_extract, contents[0])
            inner = [
                d for d in os.listdir(candidate)
                if os.path.isdir(os.path.join(candidate, d))
            ]
            if len(inner) >= 2:
                source = candidate

        os.makedirs(dest_dir, exist_ok=True)
        shutil.copytree(source, dest_dir, dirs_exist_ok=True)
        logger.info(f"Dataset copied to {dest_dir}")

        # Cleanup
        shutil.rmtree(tmp_extract, ignore_errors=True)
        if os.path.isfile(zip_path):
            os.remove(zip_path)
            logger.info(f"Removed {zip_path}")


    def create_kfold_splits(self, k=5, seed=42):
        """
        Create k stratified fold directories + an 'all' directory.
        Each fold has its own train/ and val/ subdirectories.
        The 'all' directory contains every image for final-model training.
        """
        source_dir = os.path.join(
            str(self.config.unzip_dir), _DATASET_DIR_NAME
        )
        folds_dir = os.path.join(str(self.config.unzip_dir), "folds")
        all_dir = os.path.join(str(self.config.unzip_dir), "all")

        # Skip if already created
        if os.path.exists(folds_dir) and os.path.exists(all_dir):
            logger.info("K-fold splits already exist, skipping.")
            return

        random.seed(seed)

        # Collect files per class
        class_files = {}
        for class_name in sorted(os.listdir(source_dir)):
            class_path = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            files = sorted(os.listdir(class_path))
            random.shuffle(files)
            class_files[class_name] = files

        # --- Create 'all' directory (every image, for final model) ---
        for class_name, files in class_files.items():
            dest = os.path.join(all_dir, class_name)
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copy2(
                    os.path.join(source_dir, class_name, f),
                    os.path.join(dest, f)
                )
            logger.info(f"Copied {len(files)} {class_name} images to all/")

        # --- Create k stratified folds ---
        for fold in range(1, k + 1):
            fold_train = os.path.join(folds_dir, f"fold_{fold}", "train")
            fold_val = os.path.join(folds_dir, f"fold_{fold}", "val")

            for class_name, files in class_files.items():
                os.makedirs(os.path.join(fold_train, class_name), exist_ok=True)
                os.makedirs(os.path.join(fold_val, class_name), exist_ok=True)

                n = len(files)
                fold_size = n // k
                remainder = n % k

                # Distribute remainder: first 'remainder' folds get one extra sample
                start = sum(fold_size + (1 if i < remainder else 0)
                            for i in range(fold - 1))
                end = start + fold_size + (1 if (fold - 1) < remainder else 0)

                val_files = files[start:end]
                train_files = files[:start] + files[end:]

                for f in train_files:
                    shutil.copy2(
                        os.path.join(source_dir, class_name, f),
                        os.path.join(fold_train, class_name, f)
                    )
                for f in val_files:
                    shutil.copy2(
                        os.path.join(source_dir, class_name, f),
                        os.path.join(fold_val, class_name, f)
                    )

                logger.info(
                    f"Fold {fold} - {class_name}: "
                    f"{len(train_files)} train, {len(val_files)} val"
                )

        logger.info(f"K-fold splits complete (k={k}, seed={seed})")

