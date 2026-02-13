import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

class PredictionPipeline:
    def __init__(self, filename):
        self.filename=filename
        self.model = load_model(os.path.join("model", "model.h5"))


    def predict(self):
        
        test_image = image.load_img(self.filename, target_size=(128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        pred_index = int(np.argmax(self.model.predict(test_image), axis=1))

        prediction = CLASS_NAMES[pred_index]
        return [{"image": prediction}]