from tensorflow.keras.callbacks import Callback


class LogCosineDecay(Callback):
    def __init__(self, lr_schedule, log_storage):
        super().__init__()
        self.lr_schedule = lr_schedule
        self.log_storage = log_storage

    def on_epoch_end(self, epoch, logs=None):
        steps = len(self.model.history.epoch) * len(self.model.history.history.get('loss', []))
        current_lr = float(self.lr_schedule(epoch * steps))
        self.log_storage.append(current_lr)
        print(f"Epoch {epoch + 1}: Learning Rate = {current_lr:.6f}")

