import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Chicken_Disease_Classification.entity.config_entity import TrainingConfig
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras import regularizers


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        """
        Load the updated base model and recompile with custom settings.
        Add BatchNorm + Dropout + L2 regularization to prevent overfitting.
        """
        # Load pretrained / updated base model
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # ---- Step 1: Infer number of classes ----
        df = pd.read_csv(self.config.label_csv_path)
        num_classes = df['label'].nunique()

        # ---- Step 2: Adjust last layers if mismatch ----
        if hasattr(self.model.layers[-1], "units") and self.model.layers[-1].units != num_classes:
            print(f"⚡ Adjusting last Dense layer: {self.model.layers[-1].units} ➝ {num_classes}")
            x = self.model.layers[-2].output

            # 🔹 Add Batch Normalization
            x = tf.keras.layers.BatchNormalization()(x)

            # 🔹 Add Dropout
            x = tf.keras.layers.Dropout(0.5)(x)

            # 🔹 Final Dense with L2
            output = tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
                kernel_regularizer=regularizers.l2(0.01)
            )(x)

            self.model = tf.keras.Model(inputs=self.model.input, outputs=output)

        # ---- Step 3: Re-compile ----
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print("✅ Model compiled with BatchNorm + Dropout + L2 regularization.")

    def train_valid_generator(self):
        """
        Prepare training and validation generators using the image folder + CSV.
        Includes strong augmentation to reduce overfitting.
        """
        df = pd.read_csv(self.config.label_csv_path)

        # Print dataset distribution
        print("\n📊 Class Distribution in Dataset:")
        print(df['label'].value_counts())

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = ImageDataGenerator(**datagenerator_kwargs)

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

        if self.config.params_is_augmentation:
            train_datagenerator = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],  # 🔹 brightness augmentation
                channel_shift_range=20.0,      # 🔹 color augmentation
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_dataframe(
            dataframe=df,
            directory=self.config.image_data_dir,
            x_col='images',
            y_col='label',
            subset="training",
            shuffle=True,
            class_mode='categorical',
            **dataflow_kwargs
        )

        # Print loaded dataset summary
        print(f"\n🖼️ Total training images: {self.train_generator.samples}")
        print(f"🖼️ Total validation images: {self.valid_generator.samples}")
        print(f"📂 Classes found: {len(self.train_generator.class_indices)}")
        print(f"📂 Class indices: {self.train_generator.class_indices}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save model in modern `.keras` format.
        """
        save_path = str(path).replace(".h5", ".keras")  # ✅ ensure keras format
        model.save(save_path)
        print(f"✅ Model saved at: {save_path}")

    def train(self, callback_list: list):
        """
        Train the model with generators, handle class imbalance,
        and save the best version in `.keras` format.
        """
        # ---- Compute class weights for imbalance ----
        df = pd.read_csv(self.config.label_csv_path)

        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['label'])

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(df['label_encoded']),
            y=df['label_encoded']
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print("\n📊 Computed class weights:", class_weight_dict)

        # ---- Add EarlyStopping + ReduceLROnPlateau + ModelCheckpoint + TensorBoard ----
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )

        model_checkpoint = ModelCheckpoint(
            filepath=str(self.config.trained_model_path).replace(".h5", "_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        # 🔹 TensorBoard callback
        log_dir = os.path.join("logs", time.strftime("run_%Y%m%d-%H%M%S"))
        tensorboard_cb = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        print(f"📊 TensorBoard logs saved at: {log_dir}")

        # Merge user-provided callbacks with these
        callback_list = callback_list + [early_stopping, reduce_lr, model_checkpoint, tensorboard_cb]

        # ---- Train the model ----
        # ✅ CORRECTION: Removed steps_per_epoch and validation_steps.
        # Keras will automatically determine the number of steps from the generator.
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=callback_list,
            class_weight=class_weight_dict
        )

        # ---- Save final trained model ----
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        return history