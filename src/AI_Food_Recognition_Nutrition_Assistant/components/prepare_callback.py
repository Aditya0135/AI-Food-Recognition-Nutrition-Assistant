import tensorflow as tf
import time
import os
from AI_Food_Recognition_Nutrition_Assistant.config.configuration import PrepareCallbackConfig

class PrepareCallback:
    
    def __init__(self, config: PrepareCallbackConfig):
        self.config = config
    

    @property
    def _create_tb_callback(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )
    
    def get_tb_ckpt_callbacks(self):
        return[
            self._create_ckpt_callbacks,
            self._create_tb_callback
        ]

