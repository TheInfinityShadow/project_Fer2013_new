from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt
import seaborn as sns
import numpy as np
import os


def evaluate_model(model, x_test, y_train, label_encoder, PLOTS_DIR):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_train, axis=1)

    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.show()
