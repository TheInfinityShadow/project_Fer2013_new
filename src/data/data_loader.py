from tensorflow.keras.applications.resnet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import os


def load_data(FOLDER_PATH, INPUT_SIZE, NUM_CLASSES):
    DATA_PATH = os.path.join(FOLDER_PATH, "*", "*")
    le = LabelEncoder()
    data, labels = [], []
    i = 0
    for i, path in enumerate(glob.glob(DATA_PATH)):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = preprocess_input(img)
        data.append(img)

        label = path.split("\\")[-2]
        labels.append(label)

    print(f"Total data loaded: {i}")
    data = np.array(data)
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, NUM_CLASSES)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test, le
