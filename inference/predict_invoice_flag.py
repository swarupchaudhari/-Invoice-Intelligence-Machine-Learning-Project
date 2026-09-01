import joblib
import pandas as pd


MODEL_PATH = "models/predict_flag_invoice.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice flagging classifier.
    """

    model = joblib.load(model_path)

    return model


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Input features required by the trained model.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the predicted invoice flag.
    """

    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Flag"] = model.predict(input_df)

    return input_df


if __name__ == "__main__":

    # Example prediction
    sample_data = {
        "invoice_quantity": [100, 200],
        "invoice_dollars": [5000, 15000],
        "Freight": [200, 800],
        "total_item_quantity": [100, 190],
        "total_item_dollars": [5000, 14000]
    }

    prediction = predict_invoice_flag(sample_data)

    print("\nInvoice Flag Prediction:")
    print(prediction)