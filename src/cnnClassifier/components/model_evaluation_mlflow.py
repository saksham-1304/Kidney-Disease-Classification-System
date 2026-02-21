import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    # ------------------------------------------------------------------
    #  Per-directory evaluation helper
    # ------------------------------------------------------------------
    def _evaluate_on_dir(self, model, data_dir):
        """Evaluate *model* on every image in *data_dir* and return a
        metrics dict with loss, accuracy, macro P/R/F1, and per-class
        breakdown."""
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        gen = datagen.flow_from_directory(
            directory=data_dir,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            shuffle=False
        )

        loss, accuracy = model.evaluate(gen)

        gen.reset()
        y_pred_probs = model.predict(gen)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = gen.classes
        class_names = list(gen.class_indices.keys())

        per_class = {}
        for idx, cls in enumerate(class_names):
            tp = int(np.sum((y_pred == idx) & (y_true == idx)))
            fp = int(np.sum((y_pred == idx) & (y_true != idx)))
            fn = int(np.sum((y_pred != idx) & (y_true == idx)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            per_class[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(np.sum(y_true == idx))
            }

        macro_p = float(np.mean([v["precision"] for v in per_class.values()]))
        macro_r = float(np.mean([v["recall"] for v in per_class.values()]))
        macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

        return {
            "loss": round(float(loss), 6),
            "accuracy": round(float(accuracy), 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": per_class,
            "samples": len(y_true)
        }

    # ------------------------------------------------------------------
    #  Main evaluation entry point
    # ------------------------------------------------------------------
    def evaluation(self):
        k = self.config.params_k_folds
        data_root = str(self.config.data_root)
        folds_dir = os.path.join(data_root, "folds")

        fold_results = []
        for fold in range(1, k + 1):
            logger.info(f"{'='*20} Evaluating Fold {fold}/{k} {'='*20}")
            fold_val = os.path.join(folds_dir, f"fold_{fold}", "val")
            fold_model_path = str(self.config.path_of_model).replace(
                "model.h5", f"model_fold_{fold}.h5"
            )
            model = self.load_model(fold_model_path)
            metrics = self._evaluate_on_dir(model, fold_val)
            fold_results.append(metrics)
            logger.info(
                f"Fold {fold}: loss={metrics['loss']:.6f}, "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

        # --- Aggregate across folds ---
        accs = [r["accuracy"] for r in fold_results]
        losses = [r["loss"] for r in fold_results]
        f1s = [r["macro_f1"] for r in fold_results]
        precs = [r["macro_precision"] for r in fold_results]
        recs = [r["macro_recall"] for r in fold_results]

        self.detailed_scores = {
            "k_folds": k,
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "mean_loss": round(float(np.mean(losses)), 6),
            "std_loss": round(float(np.std(losses)), 6),
            "mean_macro_precision": round(float(np.mean(precs)), 4),
            "std_macro_precision": round(float(np.std(precs)), 4),
            "mean_macro_recall": round(float(np.mean(recs)), 4),
            "std_macro_recall": round(float(np.std(recs)), 4),
            "mean_macro_f1": round(float(np.mean(f1s)), 4),
            "std_macro_f1": round(float(np.std(f1s)), 4),
            "per_fold": fold_results,
            "total_samples": sum(r["samples"] for r in fold_results)
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"K-Fold Cross-Validation Results (k={k}):")
        logger.info(f"  Accuracy:        {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        logger.info(f"  Loss:            {np.mean(losses):.6f} +/- {np.std(losses):.6f}")
        logger.info(f"  Macro F1:        {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        logger.info(f"  Macro Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
        logger.info(f"  Macro Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")

        self.save_score()

        # Store aggregates for MLflow
        self.score = [float(np.mean(losses)), float(np.mean(accs))]

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.detailed_scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        model = self.load_model(self.config.path_of_model)

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(model, "model")
