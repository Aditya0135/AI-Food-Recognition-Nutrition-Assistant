import tensorflow
from AI_Food_Recognition_Nutrition_Assistant.config.configuration import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    def get_base_model(self):
        self.model = tensorflow.keras.applications.VGG16(
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            input_shape=self.config.params_image_size,
        )
        self.save_model(path = self.config.base_model_path, model = self.model) 

    @staticmethod
    def _prepare_full_model(model,classes,freeze_all,freeze_till,learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tensorflow.keras.layers.Flatten()(model.output)
        predication = tensorflow.keras.layers.Dense(
            units=classes,
            activation='softmax'
        )(flatten_in)
        
        full_model = tensorflow.keras.models.Model(
            inputs=model.input,
            outputs=predication
        )

        full_model.compile(
            optimizer=tensorflow.keras.optimizers.SGD(learning_rate = learning_rate),
            loss=tensorflow.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path = self.config.updated_base_model_path, model = self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tensorflow.keras.Model):
        model.save(path)