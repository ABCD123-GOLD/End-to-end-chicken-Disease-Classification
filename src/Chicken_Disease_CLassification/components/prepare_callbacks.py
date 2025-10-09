import os
import time
import tensorflow as tf
from pathlib import Path
from Chicken_Disease_Classification.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def create_tb_callbacks(self):
        """Create TensorBoard callback with timestamped log directory"""
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            str(self.config.tensorboard_root_log_dir),
            f"tb_logs_at_{timestamp}"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(tb_running_log_dir, exist_ok=True)
        
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def create_ckpt_callbacks(self):
        """Create ModelCheckpoint callback"""
        # Create directory for checkpoint if it doesn't exist
        checkpoint_dir = os.path.dirname(str(self.config.checkpoint_model_filepath))
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

    def get_tb_ckpt_callbacks(self):
        """Get both TensorBoard and ModelCheckpoint callbacks"""
        return [
            self.create_tb_callbacks,
            self.create_ckpt_callbacks
        ]
