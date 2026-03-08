
def plot_accuracy_score(
    history=None, 
    y_true=None, 
    y_pred_classes=None,
    model_name="Model"):
    """
    Evaluates the model, plots training history, shows confusion matrix, and prints classification report.

    Args:
        history: Training history object (from model.fit), optional.
        model_name (str): Name of the model for titles and labeling.
    """
    try:
        if history is not None:
            plt.figure(figsize=(12, 5))

            # Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(
                history.history['accuracy'], 
                label='Train Accuracy', 
                marker='o', 
                color='red'
            )
            plt.plot(
                history.history['val_accuracy'], 
                label='Val Accuracy', 
                marker='o', 
                color='green'
            )
            plt.title(f'{model_name} - Accuracy Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error as: {e}")