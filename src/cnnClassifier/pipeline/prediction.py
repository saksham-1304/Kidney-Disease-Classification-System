import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Alphabetical order — matches flow_from_directory class indices
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = load_model(os.path.join("model", "model.h5"), compile=False)

    def predict(self):
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # must match training rescale
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(self.model.predict(test_image), axis=1)

        prediction = CLASS_NAMES[result[0]]
        return [{"image": prediction}]
