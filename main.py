import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from configs import default_config as cnf
from src.data.data_loader import load_data
from src.data.augment import augment_data
from src.models.build_model import build_model
from src.models.cbam import CBAMLayer
from src.training.learning_rate import cosineDecay
from src.training.loss import focal_loss
from src.training.callbacks import get_callbacks
from src.training.lr_scheduler import LogCosineDecay
from src.training.training import train_model
from src.utils.evaluation import evaluate_model
from src.utils.plotting import plot_history


def main():
    print("🔹 Loading data...")
    x_train, x_test, y_train, y_test, le = load_data(cnf.TRAIN_DIR, cnf.INPUT_SIZE, cnf.NUM_CLASSES)

    print("🔹 Applying data augmentation...")
    datagen = augment_data(x_train)

    print("🔹 Building model...")
    learning_rate = cosineDecay
    loss = focal_loss(cnf.FL_GAMMA, cnf.FL_ALPHA)
    cbam = CBAMLayer
    model = build_model(cnf.INPUT_SIZE, cbam, cnf.NUM_CLASSES, learning_rate, loss)

    print("🔹 Starting training...")
    lr_log = []
    log_callback = LogCosineDecay(learning_rate, lr_log)
    callbacks = get_callbacks(cnf.MODELS_CHECKPOINTS_DIR, cnf.ES_PATIENCE)
    results = train_model(model, callbacks, log_callback, datagen, x_train, x_test, y_train, y_test, cnf.BATCH_SIZE,
                          cnf.EPOCHS, cnf.MODELS_FINAL_MODEL_DIR)

    print("🔹 Evaluating model...")
    evaluate_model(model, x_test, y_test, le, cnf.RESULTS_PLOTS_DIR)

    print("🔹 Plotting history...")
    plot_history(results, cnf.RESULTS_PLOTS_DIR)


if __name__ == "__main__":
    main()
