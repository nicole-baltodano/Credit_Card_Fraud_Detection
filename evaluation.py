import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score

def plot_metrics(history):
    """Plot training and validation metrics."""
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    # Loss
    ax[0].plot(history.history['loss'], label='Train')
    ax[0].plot(history.history['val_loss'], label='Val')
    ax[0].set_title('Loss')
    ax[0].legend()

    # Recall
    ax[1].plot(history.history['recall'], label='Train')
    ax[1].plot(history.history['val_recall'], label='Val')
    ax[1].set_title('Recall')
    ax[1].legend()

    # Precision
    ax[2].plot(history.history['precision'], label='Train')
    ax[2].plot(history.history['val_precision'], label='Val')
    ax[2].set_title('Precision')
    ax[2].legend()

    plt.show()

def plot_confusion_matrix(y_test, y_pred_binary):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate the model on the test set."""
    y_pred_proba = model.predict(X_test)
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    recall = recall_score(y_test, y_pred_binary)
    print(f"Recall: {recall:.2f}")
    return y_pred_binary
