import os
import time
import tensorflow as tf
import numpy as np
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    @staticmethod
    def _enable_eager_compat():
        tf.config.run_functions_eagerly(False)

    def _compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def get_base_model(self):
        """Load a fresh copy of the compiled base model."""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )
        self._compile_model()

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    @staticmethod
    def _count_completed_epochs(history_path: str) -> int:
        if not os.path.exists(history_path):
            return 0

        with open(history_path, "r", encoding="utf-8") as file:
            line_count = sum(1 for _ in file)

        return max(0, line_count - 1)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        return f"{minutes:02d}m {secs:02d}s"

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _dataflow_kwargs(self):
        return dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

    def _get_callbacks(
        self,
        resume_model_path: str,
        history_path: str,
        completed_epochs: int
    ):
        heartbeat_state = {
            "train_start": 0.0,
            "epoch_start": 0.0,
        }

        def on_train_begin(logs=None):
            heartbeat_state["train_start"] = time.time()

        def on_epoch_begin(epoch, logs=None):
            heartbeat_state["epoch_start"] = time.time()

        def on_epoch_end(epoch, logs=None):
            epoch_seconds = time.time() - heartbeat_state["epoch_start"]
            elapsed_session = time.time() - heartbeat_state["train_start"]
            epoch_number = epoch + 1
            completed_session = max(1, epoch_number - completed_epochs)
            avg_session_epoch = elapsed_session / completed_session
            remaining_epochs = max(0, self.config.params_epochs - epoch_number)
            eta_seconds = remaining_epochs * avg_session_epoch

            logger.info(
                f"Heartbeat - epoch {epoch_number}/{self.config.params_epochs} | "
                f"epoch_time={self._format_duration(epoch_seconds)} | "
                f"elapsed={self._format_duration(elapsed_session)} | "
                f"eta={self._format_duration(eta_seconds)}"
            )

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
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=resume_model_path,
                monitor="val_loss",
                save_best_only=False,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                filename=history_path,
                append=True
            ),
            tf.keras.callbacks.LambdaCallback(
                on_train_begin=on_train_begin,
                on_epoch_begin=on_epoch_begin,
                on_epoch_end=on_epoch_end
            )
        ]

    @staticmethod
    def _compute_class_weights(generator):
        class_counts = np.bincount(generator.classes)
        non_zero = class_counts > 0
        num_classes = int(non_zero.sum())
        total_samples = int(class_counts.sum())

        class_weight = {}
        for class_idx, count in enumerate(class_counts):
            if count > 0:
                class_weight[class_idx] = total_samples / (num_classes * count)
            else:
                class_weight[class_idx] = 0.0

        return class_weight

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
        if os.path.exists(model_save_path):
            logger.info(f"Existing fold model found, skipping: {model_save_path}")
            return

        resume_model_path = model_save_path.replace(".h5", "_resume.keras")
        history_path = model_save_path.replace(".h5", "_history.csv")
        completed_epochs = self._count_completed_epochs(history_path)

        if completed_epochs >= self.config.params_epochs and os.path.exists(resume_model_path):
            logger.info(
                f"Fold already trained ({completed_epochs} epochs). "
                f"Finalizing from checkpoint: {resume_model_path}"
            )
            finalized_model = tf.keras.models.load_model(resume_model_path, compile=False)
            self._compile_model_for_model(finalized_model)
            self.save_model(path=model_save_path, model=finalized_model)
            return

        tf.keras.backend.clear_session()
        self._enable_eager_compat()

        if os.path.exists(resume_model_path) and completed_epochs > 0:
            logger.info(
                f"Resuming fold from epoch {completed_epochs}/{self.config.params_epochs}: "
                f"{resume_model_path}"
            )
            self.model = tf.keras.models.load_model(resume_model_path, compile=False)
            self._compile_model()
        else:
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

        logger.info(
            f"Training for up to {self.config.params_epochs} epochs "
            f"({train_gen.samples} train, {val_gen.samples} val images)"
        )

        class_weight = self._compute_class_weights(train_gen)
        logger.info(f"Using class weights: {class_weight}")

        self.model.fit(
            train_gen,
            epochs=self.config.params_epochs,
            initial_epoch=completed_epochs,
            validation_data=val_gen,
            callbacks=self._get_callbacks(
                resume_model_path,
                history_path,
                completed_epochs
            ),
            class_weight=class_weight
        )

        self.save_model(path=model_save_path, model=self.model)

    def _train_final(self, all_dir, model_save_path):
        """Train the final (deployable) model on ALL data with an internal
        80/20 split used only for EarlyStopping / ReduceLROnPlateau."""
        if os.path.exists(model_save_path):
            logger.info(f"Existing final model found, skipping: {model_save_path}")
            return

        resume_model_path = model_save_path.replace(".h5", "_resume.keras")
        history_path = model_save_path.replace(".h5", "_history.csv")
        completed_epochs = self._count_completed_epochs(history_path)

        if completed_epochs >= self.config.params_epochs and os.path.exists(resume_model_path):
            logger.info(
                f"Final model already trained ({completed_epochs} epochs). "
                f"Finalizing from checkpoint: {resume_model_path}"
            )
            finalized_model = tf.keras.models.load_model(resume_model_path, compile=False)
            self._compile_model_for_model(finalized_model)
            self.save_model(path=model_save_path, model=finalized_model)
            return

        tf.keras.backend.clear_session()
        self._enable_eager_compat()

        if os.path.exists(resume_model_path) and completed_epochs > 0:
            logger.info(
                f"Resuming final model from epoch {completed_epochs}/{self.config.params_epochs}: "
                f"{resume_model_path}"
            )
            self.model = tf.keras.models.load_model(resume_model_path, compile=False)
            self._compile_model()
        else:
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

        logger.info(
            f"Final model: training for up to {self.config.params_epochs} epochs "
            f"({train_gen.samples} train, {val_gen.samples} val images)"
        )

        class_weight = self._compute_class_weights(train_gen)
        logger.info(f"Using class weights (final): {class_weight}")

        self.model.fit(
            train_gen,
            epochs=self.config.params_epochs,
            initial_epoch=completed_epochs,
            validation_data=val_gen,
            callbacks=self._get_callbacks(
                resume_model_path,
                history_path,
                completed_epochs
            ),
            class_weight=class_weight
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

    def _compile_model_for_model(self, model: tf.keras.Model):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

