"""Train an educational fraud classifier using a dataset hosted on Hugging Face.

This anonymized benchmark is for learning and is not suitable for real payment
decisions without additional validation and monitoring.
"""

import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_FILE = (
    "hf://datasets/David-Egea/Creditcard-fraud-detection/creditcard.csv"
)
TARGET = "Class"
PREDICTION_THRESHOLD = 0.5


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    dataset = load_dataset(
        "csv",
        data_files=DATASET_FILE,
        split="train",
        token=os.getenv("HF_API_KEY"),
    )
    data = dataset.to_pandas()

    if TARGET not in data.columns:
        raise ValueError(f"Dataset is missing the expected target column: {TARGET}")

    features = data.drop(columns=[TARGET])
    labels = data[TARGET].astype("int8")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    fraud_count = int((y_train == 1).sum())
    legitimate_count = int((y_train == 0).sum())
    if fraud_count == 0:
        raise ValueError("Training split contains no fraud examples.")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=legitimate_count / fraud_count,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)

    fraud_probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (fraud_probabilities >= PREDICTION_THRESHOLD).astype("int8")

    print(f"Dataset rows: {len(data):,}")
    print(f"Fraud rate: {labels.mean():.3%}")
    average_precision = average_precision_score(y_test, fraud_probabilities)
    print(f"Average precision (PR-AUC): {average_precision:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, fraud_probabilities):.4f}")
    print(f"Confusion matrix at threshold {PREDICTION_THRESHOLD}:")
    print(confusion_matrix(y_test, predictions))
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["legitimate", "fraud"],
            zero_division=0,
        )
    )

    output_dir = PROJECT_DIR / "artifacts"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "fraud_xgboost.json"
    model.save_model(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
