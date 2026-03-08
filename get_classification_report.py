def get_classification_report(
        model_name, 
        y_pred, 
        y_pred_classes, 
        y_true, 
        class_names
    ):
    """
    Get and prints classification report.

    Args:
        model_name: name of the model being evaluated
        y_pred, 
        y_pred_classes, 
        y_true, 
        class_names
    """
    y_pred, y_pred_classes, y_true, class_names = None

    try:
        print(f"\n{model_name} Classification Report:")
    print(
        classification_report(
                        y_true, 
                        y_pred_classes, 
                        target_names=class_names
                        )
        )

    except Exception as e:
        print(f"Excpetion as: {e}")

    return y_pred, y_pred_classes, y_true, class_names