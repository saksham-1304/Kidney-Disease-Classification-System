from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline


os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

app = Flask(__name__)
CORS(app)

INPUT_IMAGE = "inputImage.jpg"
_classifier = None   # lazy-loaded on first prediction request


def _get_classifier():
    """Return the singleton PredictionPipeline, loading it on first call."""
    global _classifier
    if _classifier is None:
        model_path = os.path.join("model", "model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Run training first or copy model.h5 to the model/ directory."
            )
        _classifier = PredictionPipeline(INPUT_IMAGE)
    return _classifier


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    global _classifier
    _classifier = None          # invalidate cached model after retraining
    os.system("python main.py")
    return "Training done successfully!"


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        image = request.json["image"]
        decodeImage(image, INPUT_IMAGE)
        result = _get_classifier().predict()
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
