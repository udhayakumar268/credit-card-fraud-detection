
# Credit Card Fraud Detection Using Machine Learning

An end-to-end machine learning project for detecting fraudulent credit card transactions using supervised and unsupervised learning techniques, with a Random Forest model served through a FastAPI REST API.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem. The objective is to identify fraudulent transactions while minimizing unnecessary false alarms.

This project explores:
- Isolation Forest for anomaly detection
- Gaussian density estimation for anomaly detection
- Logistic Regression for supervised classification
- Random Forest for supervised classification
- Validation-based decision threshold tuning
- FastAPI for model inference

## Dataset

Source: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- Total transactions: 284,807
- Fraudulent transactions: 492
- Features: V1–V28 (anonymized PCA components), Amount, Time
- Target: Class (0 = Legitimate, 1 = Fraud)

The Time column was excluded from model training, and Amount was standardized using a scaler fitted only on the training data.

The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it in the project root.

## Project Structure

## Project Structure

```text
fraud_detection/
├── api.py
├── fraud_detection.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── outputs/
    └── figures/
        ├── class_distribution.png
        ├── gaussian_confusion_matrix.png
        ├── gaussian_log_probability.png
        ├── isolation_forest_confusion_matrix.png
        ├── logistic_regression_confusion_matrix.png
        ├── random_forest_confusion_matrix.png
        ├── random_forest_feature_importance.png
        ├── random_forest_permutation_importance.png
        └── random_forest_threshold_tuning.png
```
## Model Performance

The final selected model is Random Forest, with a decision threshold of 0.40 selected using validation-set F1 score.

### Held-Out Test Results

| Metric | Random Forest |
|---|---:|
| Precision | 0.7961 |
| Recall | 0.8367 |
| F1 Score | 0.8159 |
| True Positives | 82 |
| False Positives | 21 |
| False Negatives | 16 |
| True Negatives | 56,843 |

The model correctly detected 82 of 98 fraudulent transactions in the test set.

The test set was held out from model training and threshold selection.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

After downloading the dataset, run:

```bash
python src/train.py
```

This trains the Random Forest model, tunes the threshold on validation data, and saves the model artifacts in `models/`.

## Evaluation

```bash
python src/evaluate.py
```

This evaluates the saved model on the held-out test set and generates performance reports and a confusion matrix in `outputs/`.

## Running the FastAPI Application

Start the server:

```bash
python -m uvicorn api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### Prediction Endpoint

`POST /predict`

The request body must contain a `features` object with all 29 input features: V1–V28 and Amount.

Example response:

```json
{
  "prediction": 1,
  "result": "Fraud",
  "fraud_score": 0.8996,
  "threshold": 0.4
}
```

The fraud score is the Random Forest model's estimated class probability, not a separately calibrated probability.

## Engineering Practices

- Stratified training, validation and test splits
- Training-only scaler fitting to prevent data leakage
- Validation-based threshold selection
- Separate preprocessing, training, evaluation and inference modules
- Model and preprocessing artifacts saved with joblib
- Feature validation at inference time
- REST API built using FastAPI

## Limitations

- The dataset contains anonymized PCA features, limiting interpretability.
- Random stratified splitting does not simulate chronological deployment.
- The model has not been evaluated for concept drift or production monitoring.
- Model probabilities have not been calibrated.
- The API is a local portfolio demonstration, not a production-secured service.

## Future Improvements

- Automated API and model tests
- Docker containerization
- CI/CD with GitHub Actions
- Model monitoring and drift detection
- Probability calibration
- Cloud deployment