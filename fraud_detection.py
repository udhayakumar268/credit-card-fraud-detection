# 1. IMPORT LIBRARIES

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# 2. PROJECT PATHS

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"

FIGURES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

# 3. LOAD AND EXPLORE DATASET

df = pd.read_csv(ROOT / "creditcard.csv")

X = df.drop(columns=["Class", "Time"])
y = df["Class"]

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nClass distribution:")
print(y.value_counts())

print("\nClass percentages:")
print(y.value_counts(normalize=True) * 100)

print("\nFeature summary:")
print(df[["Time", "V1", "V2", "V3", "Amount"]].describe().round(2))

print("\nAmount statistics:")
print(df["Amount"].describe().round(2))

print("\nLargest transaction amounts:")
print(df["Amount"].nlargest(10).to_list())

# 4. CLASS DISTRIBUTION GRAPH

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Class",
)

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"],
)

plt.title("Credit Card Transaction Class Distribution")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.tight_layout()

plt.savefig(
    FIGURES / "class_distribution.png",
    dpi=150,
)

plt.close()

# 5. TRAIN / VALIDATION / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

print("\nTraining subset:", X_train_part.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)

print("\nFraud distribution:")
print("Training:", int(y_train_part.sum()))
print("Validation:", int(y_val.sum()))
print("Testing:", int(y_test.sum()))

# 6. FEATURE SCALING

X_train_part_scaled = X_train_part.copy()
X_val_scaled = X_val.copy()
X_test_part_scaled = X_test.copy()

scaler_val = StandardScaler()

X_train_part_scaled["Amount"] = scaler_val.fit_transform(
    X_train_part[["Amount"]]
)

X_val_scaled["Amount"] = scaler_val.transform(
    X_val[["Amount"]]
)

X_test_part_scaled["Amount"] = scaler_val.transform(
    X_test[["Amount"]]
)

print("\nOriginal Amount:")
print(X_train_part["Amount"].head().tolist())

print("\nScaled Amount:")
print(X_train_part_scaled["Amount"].head().tolist())

print("\nScaled mean:")
print(X_train_part_scaled["Amount"].mean())

# 7. ISOLATION FOREST


print("\n======================================")
print("ISOLATION FOREST")
print("======================================")

contamination_values = [0.002, 0.005, 0.01]

isolation_results = []

for contamination in contamination_values:

    isolation_model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )

    isolation_model.fit(X_train_part_scaled)

    predictions = isolation_model.predict(X_val_scaled)

    y_pred = (predictions == -1).astype(int)

    isolation_results.append({
        "contamination": contamination,
        "flagged": int(y_pred.sum()),
        "precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, y_pred, zero_division=0
        ),
    })

isolation_results_df = pd.DataFrame(isolation_results)

print("\nIsolation Forest Validation Results:")
print(isolation_results_df.round(4).to_string(index=False))

isolation_results_df.to_csv(
    METRICS / "isolation_forest_validation.csv",
    index=False,
)

best_isolation_row = isolation_results_df.sort_values(
    ["f1_score", "precision"],
    ascending=False,
).iloc[0]

best_contamination = float(
    best_isolation_row["contamination"]
)

print("\nSelected contamination:", best_contamination)

isolation_model = IsolationForest(
    n_estimators=100,
    contamination=best_contamination,
    random_state=42,
)

isolation_model.fit(X_train_part_scaled)

isolation_predictions = isolation_model.predict(
    X_test_part_scaled
)

isolation_test_predictions = (
    isolation_predictions == -1
).astype(int)

print("\nIsolation Forest Test Confusion Matrix:")
print(confusion_matrix(y_test, isolation_test_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        isolation_test_predictions,
        zero_division=0,
    )
)

# 8. ISOLATION FOREST CONFUSION MATRIX GRAPH

isolation_cm = confusion_matrix(
    y_test,
    isolation_test_predictions,
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    isolation_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)

plt.title("Isolation Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    FIGURES / "isolation_forest_confusion_matrix.png",
    dpi=150,
)

plt.close()

# 9. GAUSSIAN DENSITY ESTIMATION

print("\n======================================")
print("GAUSSIAN DENSITY ESTIMATION")
print("======================================")

# Train using only legitimate transactions.

X_train_part_normal = X_train_part_scaled.loc[
    y_train_part == 0
]

mu_val = X_train_part_normal.mean(
    axis=0
).to_numpy()

var_val = X_train_part_normal.var(
    axis=0,
    ddof=0,
).to_numpy()

var_val = np.maximum(var_val, 1e-9)

print("\nNormal training transactions:")
print(len(X_train_part_normal))

print("\nFirst 5 means:")
print(mu_val[:5])

print("\nFirst 5 variances:")
print(var_val[:5])

# 10. GAUSSIAN LOG PROBABILITY FUNCTION

def gaussian_log_probability(X, mu, var):

    X_array = X.to_numpy()

    log_prob_per_feature = (
        -0.5 * np.log(2 * np.pi * var)
        - ((X_array - mu) ** 2) / (2 * var)
    )

    return log_prob_per_feature.sum(axis=1)

# 11. GAUSSIAN VALIDATION SCORES

train_normal_log_prob_val = gaussian_log_probability(
    X_train_part_normal,
    mu_val,
    var_val,
)

val_log_prob = gaussian_log_probability(
    X_val_scaled,
    mu_val,
    var_val,
)

print("\nFirst 10 validation log probabilities:")
print(val_log_prob[:10])

print("\nMinimum log probability:")
print(val_log_prob.min())

print("\nMaximum log probability:")
print(val_log_prob.max())

# 12. GAUSSIAN THRESHOLD TUNING

cutoff_percentiles = [0.001, 0.005, 0.01, 0.02]

gaussian_results = []

for percentile in cutoff_percentiles:

    threshold_val = np.quantile(
        train_normal_log_prob_val,
        percentile,
    )

    val_predictions = (
        val_log_prob < threshold_val
    ).astype(int)

    gaussian_results.append({
        "percentile": percentile,
        "threshold": threshold_val,
        "flagged": int(val_predictions.sum()),
        "precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, val_predictions, zero_division=0
        ),
    })

gaussian_results_df = pd.DataFrame(gaussian_results)

print("\nGaussian Validation Results:")
print(gaussian_results_df.round(4).to_string(index=False))

gaussian_results_df.to_csv(
    METRICS / "gaussian_validation.csv",
    index=False,
)

# Keep the original selected 0.005 quantile.

final_threshold = np.quantile(
    train_normal_log_prob_val,
    0.005,
)

print("\nSelected Gaussian threshold:")
print(final_threshold)

# 13. GAUSSIAN FINAL TEST EVALUATION

test_log_prob_final = gaussian_log_probability(
    X_test_part_scaled,
    mu_val,
    var_val,
)

gaussian_test_predictions = (
    test_log_prob_final < final_threshold
).astype(int)

print("\nGaussian Test Confusion Matrix:")
print(confusion_matrix(y_test, gaussian_test_predictions))

print("\nGaussian Precision:")
print(
    precision_score(
        y_test,
        gaussian_test_predictions,
        zero_division=0,
    )
)

print("\nGaussian Recall:")
print(
    recall_score(
        y_test,
        gaussian_test_predictions,
        zero_division=0,
    )
)

print("\nGaussian F1 Score:")
print(
    f1_score(
        y_test,
        gaussian_test_predictions,
        zero_division=0,
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        gaussian_test_predictions,
        zero_division=0,
    )
)

# 14. GAUSSIAN LOG PROBABILITY VISUALIZATION

# Use validation scores for visualization.
# Extreme values are excluded from the displayed range
# so the distribution remains readable.
# This does not change model scores or predictions.

normal_scores = val_log_prob[
    y_val.to_numpy() == 0
]

fraud_scores = val_log_prob[
    y_val.to_numpy() == 1
]

plot_min, plot_max = np.percentile(
    val_log_prob,
    [0.5, 99.5],
)

plt.figure(figsize=(11, 6))

bins = np.linspace(
    plot_min,
    plot_max,
    80,
)

plt.hist(
    normal_scores,
    bins=bins,
    alpha=0.6,
    density=True,
    label="Legitimate Transactions",
    color="steelblue",
)

plt.hist(
    fraud_scores,
    bins=bins,
    alpha=0.6,
    density=True,
    label="Fraud Transactions",
    color="tomato",
)

plt.axvline(
    final_threshold,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Threshold: {final_threshold:.2f}",
)

plt.xlim(plot_min, plot_max)

plt.title("Gaussian Log Probability Distribution")
plt.xlabel("Log Probability")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES / "gaussian_log_probability.png",
    dpi=150,
)

plt.close()

# 15. GAUSSIAN CONFUSION MATRIX GRAPH

gaussian_cm = confusion_matrix(
    y_test,
    gaussian_test_predictions,
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    gaussian_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)

plt.title("Gaussian Density - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    FIGURES / "gaussian_confusion_matrix.png",
    dpi=150,
)

plt.close()

# 16. LOGISTIC REGRESSION

print("\n======================================")
print("LOGISTIC REGRESSION")
print("======================================")

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)

logistic_model.fit(
    X_train_part_scaled,
    y_train_part,
)

logistic_val_prob = logistic_model.predict_proba(
    X_val_scaled
)[:, 1]

logistic_thresholds = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9,
    0.92, 0.94, 0.95, 0.96,
    0.97, 0.98, 0.99,
    0.992, 0.994, 0.996,
    0.998, 0.999, 0.9995,
]

logistic_results = []

for threshold in logistic_thresholds:

    val_predictions = (
        logistic_val_prob >= threshold
    ).astype(int)

    logistic_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, val_predictions, zero_division=0
        ),
        "flagged_transactions": int(
            val_predictions.sum()
        ),
    })

logistic_validation_results = pd.DataFrame(
    logistic_results
)

print("\nLogistic Regression Validation Results:")
print(
    logistic_validation_results.to_string(
        index=False
    )
)

logistic_validation_results.to_csv(
    METRICS / "logistic_regression_validation.csv",
    index=False,
)

best_row = logistic_validation_results.sort_values(
    ["f1_score", "precision"],
    ascending=False,
).iloc[0]

best_logistic_threshold = float(
    best_row["threshold"]
)

print("\nBest Logistic Regression Threshold:")
print(best_logistic_threshold)

print("\nBest Validation F1:")
print(best_row["f1_score"])

# 17. LOGISTIC REGRESSION FINAL TEST

logistic_test_prob = logistic_model.predict_proba(
    X_test_part_scaled
)[:, 1]

logistic_test_predictions = (
    logistic_test_prob >= best_logistic_threshold
).astype(int)

logistic_precision = precision_score(
    y_test,
    logistic_test_predictions,
    zero_division=0,
)

logistic_recall = recall_score(
    y_test,
    logistic_test_predictions,
    zero_division=0,
)

logistic_f1 = f1_score(
    y_test,
    logistic_test_predictions,
    zero_division=0,
)

print("\nLogistic Regression Test Results:")
print("Precision:", round(logistic_precision, 4))
print("Recall:", round(logistic_recall, 4))
print("F1 Score:", round(logistic_f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_test_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_test_predictions,
        zero_division=0,
    )
)

# 18. LOGISTIC REGRESSION CONFUSION MATRIX GRAPH

logistic_cm = confusion_matrix(
    y_test,
    logistic_test_predictions,
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    logistic_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    FIGURES / "logistic_regression_confusion_matrix.png",
    dpi=150,
)

plt.close()

# 19. RANDOM FOREST

print("\n======================================")
print("RANDOM FOREST")
print("======================================")

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

random_forest_model.fit(
    X_train_part_scaled,
    y_train_part,
)

rf_val_prob = random_forest_model.predict_proba(
    X_val_scaled
)[:, 1]

rf_thresholds = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95,
]

rf_results = []

for threshold in rf_thresholds:

    rf_val_predictions = (
        rf_val_prob >= threshold
    ).astype(int)

    rf_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, rf_val_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, rf_val_predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, rf_val_predictions, zero_division=0
        ),
        "flagged_transactions": int(
            rf_val_predictions.sum()
        ),
    })

rf_validation_results = pd.DataFrame(rf_results)

print("\nRandom Forest Validation Results:")
print(
    rf_validation_results.to_string(
        index=False
    )
)

rf_validation_results.to_csv(
    METRICS / "random_forest_validation.csv",
    index=False,
)

best_rf_row = rf_validation_results.sort_values(
    ["f1_score", "precision"],
    ascending=False,
).iloc[0]

best_rf_threshold = float(
    best_rf_row["threshold"]
)

print("\nBest Random Forest Threshold:")
print(best_rf_threshold)

print("\nBest Validation F1:")
print(best_rf_row["f1_score"])

# 20. RANDOM FOREST FINAL TEST

rf_test_prob = random_forest_model.predict_proba(
    X_test_part_scaled
)[:, 1]

rf_test_predictions = (
    rf_test_prob >= best_rf_threshold
).astype(int)

rf_precision = precision_score(
    y_test,
    rf_test_predictions,
    zero_division=0,
)

rf_recall = recall_score(
    y_test,
    rf_test_predictions,
    zero_division=0,
)

rf_f1 = f1_score(
    y_test,
    rf_test_predictions,
    zero_division=0,
)

print("\nRandom Forest Test Results:")
print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1 Score:", round(rf_f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_test_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_test_predictions,
        zero_division=0,
    )
)

# 21. RANDOM FOREST CONFUSION MATRIX GRAPH

rf_cm = confusion_matrix(
    y_test,
    rf_test_predictions,
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    FIGURES / "random_forest_confusion_matrix.png",
    dpi=150,
)

plt.close()

# 22. RANDOM FOREST THRESHOLD TUNING GRAPH

plt.figure(figsize=(9, 5))

plt.plot(
    rf_validation_results["threshold"],
    rf_validation_results["precision"],
    marker="o",
    label="Precision",
)

plt.plot(
    rf_validation_results["threshold"],
    rf_validation_results["recall"],
    marker="o",
    label="Recall",
)

plt.plot(
    rf_validation_results["threshold"],
    rf_validation_results["f1_score"],
    marker="o",
    label="F1 Score",
)

plt.axvline(
    best_rf_threshold,
    color="black",
    linestyle="--",
    label=f"Selected Threshold: {best_rf_threshold:.2f}",
)

plt.title("Random Forest - Threshold Tuning")
plt.xlabel("Decision Threshold")
plt.ylabel("Validation Score")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES / "random_forest_threshold_tuning.png",
    dpi=150,
)

plt.close()

# 23. RANDOM FOREST FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "Feature": X_train_part_scaled.columns,
    "Importance": random_forest_model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False,
)

print("\nTop 10 Random Forest Features:")
print(feature_importance.head(10).to_string(index=False))

plt.figure(figsize=(9, 6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature",
)

plt.title("Random Forest - Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    FIGURES / "random_forest_feature_importance.png",
    dpi=150,
)

plt.close()

# 24. RANDOM FOREST PERMUTATION IMPORTANCE

print("\nCalculating Permutation Importance...")

permutation_result = permutation_importance(
    random_forest_model,
    X_val_scaled,
    y_val,
    scoring="average_precision",
    n_repeats=3,
    random_state=42,
    n_jobs=-1,
)

permutation_df = pd.DataFrame({
    "Feature": X_val_scaled.columns,
    "Importance": permutation_result.importances_mean,
})

permutation_df = permutation_df.sort_values(
    by="Importance",
    ascending=False,
)

print("\nTop 10 Features - Permutation Importance:")
print(permutation_df.head(10).to_string(index=False))

plt.figure(figsize=(9, 6))

sns.barplot(
    data=permutation_df.head(10),
    x="Importance",
    y="Feature",
)

plt.title("Random Forest - Permutation Importance")
plt.xlabel("Decrease in Average Precision")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    FIGURES / "random_forest_permutation_importance.png",
    dpi=150,
)

plt.close()

# 25. FINAL MODEL COMPARISON

def model_metrics(name, y_true, predictions):

    return {
        "Model": name,
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


comparison_results = [
    model_metrics(
        "Isolation Forest",
        y_test,
        isolation_test_predictions,
    ),
    model_metrics(
        "Gaussian Density",
        y_test,
        gaussian_test_predictions,
    ),
    model_metrics(
        "Logistic Regression",
        y_test,
        logistic_test_predictions,
    ),
    model_metrics(
        "Random Forest",
        y_test,
        rf_test_predictions,
    ),
]

comparison_df = pd.DataFrame(comparison_results)

print("\n======================================")
print("FINAL MODEL COMPARISON")
print("======================================")

print(
    comparison_df.round(4).to_string(index=False)
)

comparison_df.to_csv(
    METRICS / "model_comparison.csv",
    index=False,
)

print("\nAll experiments completed!")
print("Graphs saved to:", FIGURES)
print("Metrics saved to:", METRICS)
