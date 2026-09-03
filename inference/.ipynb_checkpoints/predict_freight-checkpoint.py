import joblib
import pandas as pd


MODEL_PATH = "models/predict_freight_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """

    model = joblib.load(model_path)

    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Input features required by the trained model.

    Returns
    -------
    pd.DataFrame
        DataFrame with predicted freight cost.
    """

    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Freight"] = (
        model.predict(input_df).round(2)
    )

    return input_df


if __name__ == "__main__":

    # Example inference run
    sample_data = {
        "Dollars": [18500, 9000]
    }

    prediction = predict_freight_cost(sample_data)

    print(prediction)