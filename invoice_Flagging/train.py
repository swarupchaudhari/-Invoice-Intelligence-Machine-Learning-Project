from data_preprocessing import (
    load_invoice_data,
    split_data,
    scale_features,
    apply_labels
)
from modeling_evaluation import train_random_forest, evaluate_classifier
import joblib

from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features
)

from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier
)

import joblib


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    # -------------------------
    # Load data
    # -------------------------
    df = load_invoice_data()

    print("Data loaded successfully!")
    print(f"Rows: {len(df)}")

    # -------------------------
    # Create invoice risk label
    # -------------------------
    df = apply_labels(df)

    print("\nInvoice Risk Distribution:")
    print(df[TARGET].value_counts())

    # -------------------------
    # Prepare data
    # -------------------------
    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    # -------------------------
    # Scale features
    # -------------------------
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        "models/scaler.pkl"
    )

    # -------------------------
    # Train Random Forest
    # -------------------------
    print("\nTraining Random Forest...")

    grid_search = train_random_forest(
        X_train_scaled,
        y_train
    )

    # -------------------------
    # Best parameters
    # -------------------------
    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV F1 Score:")
    print(grid_search.best_score_)

    # -------------------------
    # Evaluate model
    # -------------------------
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # -------------------------
    # Save model
    # -------------------------
    joblib.dump(
        grid_search.best_estimator_,
        "models/predict_flag_invoice.pkl"
    )

    print(
        "\nBest model saved to: "
        "models/predict_flag_invoice.pkl"
    )


if __name__ == "__main__":
    main()