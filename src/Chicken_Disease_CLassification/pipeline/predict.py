import numpy as np
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    # --- THIS LINE IS THE FIX ---
    # It now accepts the 'model' object from app.py
    def predict(self, model):
        """
        This method now accepts the loaded model as an argument.
        """
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # It now uses the 'model' that was passed in, instead of loading a new one.
        result = np.argmax(model.predict(test_image), axis=1)
        
        print(f"Prediction result index: {result}")

        # Class mapping for the final prediction
        class_indices = {
            'Coccidiosis': 0, 
            'Healthy': 1, 
            'New Castle Disease': 2, 
            'Salmonella': 3
        }
        
        # Create an ordered list of class names
        prediction_classes = sorted(class_indices, key=class_indices.get)

        # Get the prediction name from the result index
        prediction = prediction_classes[result[0]]

        return prediction

