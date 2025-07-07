from sklearn.utils import class_weight
import numpy as np
import os


def train_model(model, callbacks, LogCosineDecay, datagen, x_train, x_test, y_train, y_test, BATCH_SIZE, EPOCHS,
                final_model_path):
    callbacks = [*callbacks, LogCosineDecay]

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    weights = dict(enumerate(weights))

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=weights
    )

    final_model = os.path.join(final_model_path, "final_model.keras")
    model.save(final_model)

    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")
    return history
