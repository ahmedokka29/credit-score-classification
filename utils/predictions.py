import numpy as np
import joblib


def load_scaler():
    """Load the trained MinMaxScaler."""
    scaler = joblib.load("models/min_max_scaler.joblib")
    return scaler


def load_model():
    """
    Load the trained model.
    """

    saved_data = joblib.load("models/catboost_credit_model.pkl")
    loaded_model = saved_data['model']
    loaded_thresholds = saved_data['thresholds']
    # model = joblib.load("models/best_rf_model.joblib")
    # return model
    return loaded_model, loaded_thresholds


def predict_with_thresholds_catboost(model, X_test, thresholds):
    """
    Make predictions using custom thresholds for CatBoost
    """
    y_proba = model.predict_proba(X_test)
    predictions = np.zeros(X_test.shape[0])
    class_names = list(thresholds.keys())

    # For each sample, assign class with highest probability above threshold
    for i in range(X_test.shape[0]):
        class_scores = []
        for class_idx, class_name in enumerate(class_names):
            prob = y_proba[i, class_idx]
            threshold = thresholds[class_name]['threshold']
            if prob >= threshold:
                class_scores.append((class_name, prob))

        if class_scores:
            # Choose class with highest probability among those above threshold
            predictions[i] = max(class_scores, key=lambda x: x[1])[0]
        else:
            # If no class above threshold, choose highest probability
            predictions[i] = class_names[np.argmax(y_proba[i])]

    return predictions.astype(int)
