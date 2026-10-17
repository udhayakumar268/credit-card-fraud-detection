
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocess import load_and_preprocess_data


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def train_model():

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = (
        load_and_preprocess_data()
    )

    # 2. Train Random Forest
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 3. Predict validation scores
    val_prob = model.predict_proba(X_val)[:, 1]

    # 4. Tune threshold using validation data
    thresholds = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95
    ]

    results = []

    for threshold in thresholds:

        predictions = (val_prob >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_val, predictions, zero_division=0
            )
        })

    results_df = pd.DataFrame(results)

    best_row = results_df.sort_values(
        ["f1_score", "precision"],
        ascending=False
    ).iloc[0]

    best_threshold = float(best_row["threshold"])

    print("\nValidation Results:")
    print(results_df.to_string(index=False))

    print("\nBest threshold:", best_threshold)
    print("Best validation F1:", best_row["f1_score"])

    # 5. Save model and preprocessing artifacts
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "threshold": best_threshold,
            "feature_names": X_train.columns.tolist()
        },
        MODEL_DIR / "fraud_model.joblib"
    )

    results_df.to_csv(
        OUTPUT_DIR / "random_forest_validation.csv",
        index=False
    )

    print("\nModel saved successfully!")

    return model, best_threshold


if __name__ == "__main__":
    train_model()
