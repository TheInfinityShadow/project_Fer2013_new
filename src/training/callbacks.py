from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os


def get_callbacks(checkpoint_path, patience):
    best_model = os.path.join(checkpoint_path, "best_model.keras")
    return [
        ModelCheckpoint(best_model, save_best_only=True, monitor="val_loss", mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
    ]
