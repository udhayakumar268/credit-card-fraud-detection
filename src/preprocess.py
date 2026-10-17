
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():

    # Locate dataset from project root
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "creditcard.csv"

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Separate features and target
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    # First split: 80% development, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Second split: 80% training, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    # Scale Amount using training data only
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train["Amount"] = scaler.fit_transform(
        X_train[["Amount"]]
    )

    X_val["Amount"] = scaler.transform(
        X_val[["Amount"]]
    )

    X_test["Amount"] = scaler.transform(
        X_test[["Amount"]]
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler
    )


if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = (
        load_and_preprocess_data()
    )

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    print("\nFraud cases:")
    print("Training:", int(y_train.sum()))
    print("Validation:", int(y_val.sum()))
    print("Testing:", int(y_test.sum()))
