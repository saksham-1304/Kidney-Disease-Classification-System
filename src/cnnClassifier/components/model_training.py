import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """Load a fresh copy of the compiled base model."""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _dataflow_kwargs(self):
        return dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

    def _get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

    def _get_train_datagen(self):
        """Return an ImageDataGenerator (with or without augmentation)."""
        if self.config.params_is_augmentation:
            return tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
            )
        return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # ------------------------------------------------------------------
    #  Training helpers
    # ------------------------------------------------------------------
    def _train_fold(self, train_dir, val_dir, model_save_path):
        """Train on one k-fold split (separate train / val directories)."""
        tf.keras.backend.clear_session()
        self.get_base_model()

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        val_gen = val_datagen.flow_from_directory(
            directory=val_dir,
            shuffle=False,
            seed=42,
            **self._dataflow_kwargs()
        )

        train_gen = self._get_train_datagen().flow_from_directory(
            directory=train_dir,
            shuffle=True,
            seed=42,
            **self._dataflow_kwargs()
        )

        steps = train_gen.samples // train_gen.batch_size
        val_steps = val_gen.samples // val_gen.batch_size

        logger.info(
            f"Training for up to {self.config.params_epochs} epochs "
            f"({train_gen.samples} train, {val_gen.samples} val images)"
        )

        self.model.fit(
            train_gen,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps,
            validation_steps=val_steps,
            validation_data=val_gen,
            callbacks=self._get_callbacks()
        )

        self.save_model(path=model_save_path, model=self.model)

    def _train_final(self, all_dir, model_save_path):
        """Train the final (deployable) model on ALL data with an internal
        80/20 split used only for EarlyStopping / ReduceLROnPlateau."""
        tf.keras.backend.clear_session()
        self.get_base_model()

        split_kwargs = dict(rescale=1./255, validation_split=0.20)

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **split_kwargs
        )
        val_gen = val_datagen.flow_from_directory(
            directory=all_dir,
            subset="validation",
            shuffle=False,
            seed=42,
            **self._dataflow_kwargs()
        )

        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **split_kwargs
            )
        else:
            train_datagen = val_datagen

        train_gen = train_datagen.flow_from_directory(
            directory=all_dir,
            subset="training",
            shuffle=True,
            seed=42,
            **self._dataflow_kwargs()
        )

        steps = train_gen.samples // train_gen.batch_size
        val_steps = val_gen.samples // val_gen.batch_size

        logger.info(
            f"Final model: training for up to {self.config.params_epochs} epochs "
            f"({train_gen.samples} train, {val_gen.samples} val images)"
        )

        self.model.fit(
            train_gen,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps,
            validation_steps=val_steps,
            validation_data=val_gen,
            callbacks=self._get_callbacks()
        )

        self.save_model(path=model_save_path, model=self.model)

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------
    def train(self):
        """Run k-fold training + final model on all data."""
        k = self.config.params_k_folds
        data_root = str(self.config.training_data)
        folds_dir = os.path.join(data_root, "folds")
        all_dir = os.path.join(data_root, "all")

        # --- K fold models ---
        for fold in range(1, k + 1):
            logger.info(f"{'='*20} Fold {fold}/{k} {'='*20}")
            fold_train = os.path.join(folds_dir, f"fold_{fold}", "train")
            fold_val = os.path.join(folds_dir, f"fold_{fold}", "val")
            fold_model_path = str(self.config.trained_model_path).replace(
                "model.h5", f"model_fold_{fold}.h5"
            )
            self._train_fold(fold_train, fold_val, fold_model_path)
            logger.info(f"Fold {fold} model saved to {fold_model_path}")

        # --- Final model on ALL data (for deployment) ---
        logger.info(f"{'='*20} Final Model (all data) {'='*20}")
        self._train_final(all_dir, str(self.config.trained_model_path))
        logger.info(f"Final model saved to {self.config.trained_model_path}")

