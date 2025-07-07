import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----Paths-----

# base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# configs paths
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")

# data paths
DARA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DARA_DIR, "train")
TEST_DIR = os.path.join(DARA_DIR, "test")

# models paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODELS_CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
MODELS_FINAL_MODEL_DIR = os.path.join(MODELS_DIR, "final")

# results paths
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
RESULTS_METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
RESULTS_PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# src paths
SRC_DIR = os.path.join(BASE_DIR, "src")
SRC_DATA_DIR = os.path.join(SRC_DIR, "data")
SRC_INFERENCE_DIR = os.path.join(SRC_DIR, "inference")
SRC_MODELS_DIR = os.path.join(SRC_DIR, "models")
SRC_TRAINING_DIR = os.path.join(SRC_DIR, "training")
SRC_UTILS_DIR = os.path.join(SRC_DIR, "utils")


# -----Parameters-----


INPUT_SIZE = 0
BATCH_SIZE = 0
EPOCHS = 0
NUM_CLASSES = 0

# focal loss
FL_GAMMA = 0
FL_ALPHA = 0

# cosine decay
LR_INITIAL = 0
LR_DECAY_STEPS = 0
LR_ALPHA = 0

# early stopping
ES_PATIENCE = 5
