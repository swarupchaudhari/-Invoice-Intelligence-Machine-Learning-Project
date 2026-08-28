from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model."""

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


    def train_decision_tree(X_train, y_train, max_depth=5):
    """Train Decision Tree Regression model."""

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


    def train_random_forest(X_train, y_train, max_depth=6):
    """Train Random Forest Regression model."""

    model = RandomForestRegressor(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


    import numpy as np

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate regression model and return metrics."""

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    mse = mean_squared_error(y_test, preds)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, preds)

    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    return {
        "model_name": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
