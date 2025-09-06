import joblib
import pickle
import pandas as pd
import numpy as np


def load_model_artifacts(timestamp="{timestamp}"):
    """
    Load all model artifacts for prediction
    """
    best_model = joblib.load(
        f"model_artifacts/best_model_{best_model_name.replace(' ', '_').lower()}_{{timestamp}}.joblib")
    scaler = joblib.load(
        f"model_artifacts/feature_scaler_{{timestamp}}.joblib")

    with open(f"model_artifacts/model_metadata_{{timestamp}}.pkl", 'rb') as f:
        metadata = pickle.load(f)

    return best_model, scaler, metadata


def predict_credit_score(input_data, model=None, scaler=None, metadata=None, timestamp="{timestamp}"):
    """
    Make credit score predictions on new data

    Args:
        input_data: DataFrame with same structure as training data (without target)
        model: Trained model (will load if None)
        scaler: Fitted scaler (will load if None)
        metadata: Model metadata (will load if None)
        timestamp: Model timestamp (default: {timestamp})

    Returns:
        List of dictionaries with predictions and probabilities
    """
    # Load artifacts if not provided
    if model is None or scaler is None or metadata is None:
        model, scaler, metadata = load_model_artifacts(timestamp)

    # Prepare input data
    feature_names = metadata['feature_names']
    numerical_columns = metadata['numerical_columns']
    class_names = metadata['class_names']

    # Ensure correct feature order
    input_processed = input_data[feature_names].copy()

    # Scale numerical features
    input_processed[numerical_columns] = scaler.transform(
        input_processed[numerical_columns])

    # Make predictions
    predictions = model.predict(input_processed)
    probabilities = model.predict_proba(input_processed)

    # Format results
    results = []
    for i in range(len(predictions)):
        result = {{
            'predicted_class': int(predictions[i]),
            'predicted_label': class_names[predictions[i]],
            'confidence': float(probabilities[i].max()),
            'probabilities': {{
                class_names[0]: float(probabilities[i][0]),
                class_names[1]: float(probabilities[i][1]),
                class_names[2]: float(probabilities[i][2])
            }}
        }}
        results.append(result)

    # Apply threshold tuning if optimal thresholds are available in metadata
    if metadata and 'optimal_thresholds' in metadata and metadata['optimal_thresholds']:
        tuned_predictions = []
        for i in range(len(predictions)):
            class_scores = []
            model_thresholds = metadata['optimal_thresholds'].get(
                metadata['best_model_name'])
            if model_thresholds:
                for class_idx, class_name in enumerate(class_names):
                    prob = probabilities[i][class_idx]
                    # Ensure class_id is string for dictionary lookup
                    threshold_data = model_thresholds.get(str(class_idx))
                    if threshold_data:
                        threshold = threshold_data['threshold']
                        if prob >= threshold:
                            class_scores.append((class_idx, prob))

                if class_scores:
                    # Choose class with highest probability among those above threshold
                    tuned_predictions.append(
                        max(class_scores, key=lambda x: x[1])[0])
                else:
                    # If no class above threshold, choose highest probability
                    tuned_predictions.append(np.argmax(probabilities[i]))
            else:
                # If no thresholds for this model, use original prediction
                tuned_predictions.append(predictions[i])

        # Update results with tuned predictions and recalculate confidence/probabilities based on original probs
        for i in range(len(tuned_predictions)):
            results[i]['predicted_class'] = int(tuned_predictions[i])
            results[i]['predicted_label'] = class_names[tuned_predictions[i]]
            # Confidence is still the probability of the predicted class from original model
            results[i]['confidence'] = float(
                probabilities[i][tuned_predictions[i]])

    return results
