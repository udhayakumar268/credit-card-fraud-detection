
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"


# Load saved model and preprocessing artifacts
artifacts = joblib.load(MODEL_PATH)

model = artifacts["model"]
scaler = artifacts["scaler"]
threshold = artifacts["threshold"]
feature_names = artifacts["feature_names"]


def predict_fraud(transaction):

    # Convert transaction dictionary into DataFrame
    input_data = pd.DataFrame([transaction])

    # Check for missing or unexpected features
    missing = set(feature_names) - set(input_data.columns)
    extra = set(input_data.columns) - set(feature_names)

    if missing or extra:
        raise ValueError(
            f"Missing features: {sorted(missing)}; "
            f"Unexpected features: {sorted(extra)}"
        )

    # Ensure the correct feature order
    input_data = input_data[feature_names].copy()

    # Apply the saved scaler to Amount
    input_data["Amount"] = scaler.transform(
        input_data[["Amount"]]
    )

    # Predict fraud score
    fraud_score = float(
        model.predict_proba(input_data)[0, 1]
    )

    # Apply the saved threshold
    prediction = int(fraud_score >= threshold)

    return {
        "prediction": prediction,
        "result": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_score": round(fraud_score, 4),
        "threshold": threshold
    }


if __name__ == "__main__":

    # Test using an example transaction from the dataset
    df = pd.read_csv(PROJECT_ROOT / "creditcard.csv")

    example_transaction = df.drop(
        columns=["Class", "Time"]
    ).iloc[0].to_dict()

    result = predict_fraud(example_transaction)

    print("\nTransaction Prediction:")
    print(result)
