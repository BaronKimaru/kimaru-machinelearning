def evaluate_loss_accuracy_scores(model, test_generator, history=None, model_name="Model"):
    """
    Evaluates the model, plots training history, shows confusion matrix, and prints classification report.

    Args:
        model: Trained Keras modele.g. VGG16, Xceptron
        test_generator: Test data generator.
        history: Training history object (from model.fit), optional.
        model_name (str): Name of the model for titles and labeling.
    """
    y_pred, y_pred_classes, y_true, class_names = None

    try:
        loss, accuracy = model.evaluate(test_generator, verbose=0)
        print(f"\n{model_name} Evaluation:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        y_pred = model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        class_names = list(test_generator.class_indices.keys())

    except Exception as e:
        print(f"Excpetion as: {e}")

    return y_pred, y_pred_classes, y_true, class_names