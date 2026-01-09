import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

class PredictionPipeline:
    def __init__(self, filename):
        self.filename=filename

    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        pred_index = np.argmax(model.predict(test_image), axis=1)

        prediction = CLASS_NAMES[pred_index]
        return [{"image": prediction}]