
def evaluate_and_visualize_model(model, test_generator, history=None, model_name="Model"):
    """
    Evaluates the model, plots training history, shows confusion matrix, and prints classification report.

    Args:
        model: Trained Keras model.
        test_generator: Test data generator.
        history: Training history object (from model.fit), optional.
        model_name (str): Name of the model for titles and labeling.
    """
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    print(f"\n{model_name} Evaluation:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())


    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))


    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


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

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o', color='red')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o', color='green')
        plt.title(f'{model_name} - Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()