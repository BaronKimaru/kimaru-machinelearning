
def plot_loss_score(
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

            # Loss
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss', marker='o', color='blue')
            plt.plot(history.history['val_loss'], label='Val Loss', marker='o', color='orange')
            plt.title(f'{model_name} - Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

    except Exception as e:
        print(f"Error as: {e}")