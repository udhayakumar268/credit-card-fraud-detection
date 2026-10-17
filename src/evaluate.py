
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from preprocess import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def evaluate_model():

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load saved model and threshold
    artifacts = joblib.load(MODEL_PATH)

    model = artifacts["model"]
    threshold = artifacts["threshold"]

    print("\nLoaded saved Random Forest model.")
    print("Selected threshold:", threshold)

    # 2. Load the same reproducible data split
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler
    ) = load_and_preprocess_data()

    # 3. Predict test fraud scores
    test_prob = model.predict_proba(X_test)[:, 1]

    predictions = (test_prob >= threshold).astype(int)

    # 4. Evaluate
    cm = confusion_matrix(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\nRandom Forest - Final Test Results")
    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    # 5. Save confusion matrix chart
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )

    plt.title("Random Forest - Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "random_forest_confusion_matrix.png",
        dpi=150
    )
    plt.close()

    # 6. Save metrics
    metrics_df = pd.DataFrame([{
        "model": "Random Forest",
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }])

    metrics_df.to_csv(
        OUTPUT_DIR / "random_forest_test_metrics.csv",
        index=False
    )

    print("\nEvaluation completed. Results saved to outputs/")


if __name__ == "__main__":
    evaluate_model()
