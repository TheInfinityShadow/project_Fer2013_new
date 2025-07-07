from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras import models, layers, optimizers


def build_model(INPUT_SIZE, CBAMLayer, NUM_CLASSES, learning_rate, loss):
    base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

    for layer in base_model.layers[:-50]:
        layer.trainable = False

    inputs = layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = base_model(inputs, training=False)
    x = CBAMLayer()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2048, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"]
    )

    return model
