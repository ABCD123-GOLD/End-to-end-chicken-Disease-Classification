import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import os
from Chicken_Disease_Classification.config import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = tf.keras.models.load_model(self.config.path_of_model)
        self.valid_generator = None
        self.accuracy = None
        self.loss = None

    def _valid_generator(self):
        df = pd.read_csv(self.config.label_csv_path)

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_dataframe(
            dataframe=df,
            directory=self.config.image_data_dir,
            x_col='images',
            y_col='label',
            subset="validation",
            shuffle=False,
            class_mode='categorical',
            **dataflow_kwargs
        )

        print(f"Validation generator created with {self.valid_generator.samples} samples")

    def run_evaluation(self):
        if self.valid_generator is None:
            self._valid_generator()

        # Get loss and accuracy from Keras
        self.loss, keras_accuracy = self.model.evaluate(self.valid_generator, verbose=0)

        # Manual accuracy (optional, for comparison)
        predictions = self.model.predict(self.valid_generator)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = self.valid_generator.classes
        self.accuracy = accuracy_score(true_labels, predicted_labels)

        print(f"Evaluation Accuracy (manual): {self.accuracy:.4f}")
        print(f"Evaluation Loss: {self.loss:.4f}")

    def save_score(self):
        score_dir = "artifacts/evaluation"
        os.makedirs(score_dir, exist_ok=True)

        score_path = os.path.join(score_dir, "score.txt")
        with open(score_path, "w") as f:
            f.write(f"Loss: {self.loss:.4f}\n")
            f.write(f"Accuracy: {self.accuracy:.4f}\n")

        print(f"Score saved to {score_path}")