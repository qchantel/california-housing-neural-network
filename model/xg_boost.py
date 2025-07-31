import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

def create_xgboost_model():
    """Create an XGBoost regression model with default parameters."""
    model = xgb.XGBRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=12,
        objective='reg:squarederror'
    )
    return model

def train_xgboost_model(model, X_train, y_train):
    """Train the XGBoost model on the training data."""
    model.fit(X_train, y_train)
    return model

def rmse_xgboost_model(model, X_test, y_test):
    """Calculate RMSE for the XGBoost model."""
    xgb_predictions = model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    return xgb_rmse

def evaluate_xgboost_model(model, X_test, y_test):
    """Calculate R² score for the XGBoost model."""
    xgb_predictions = model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_predictions)
    return xgb_r2

def xgboost_r2_score(X_train, y_train, X_test, y_test):
    """Complete XGBoost evaluation pipeline returning R² and RMSE scores."""
    model = create_xgboost_model()
    model = train_xgboost_model(model, X_train, y_train)
    xgb_r2 = evaluate_xgboost_model(model, X_test, y_test)
    xgb_rmse = rmse_xgboost_model(model, X_test, y_test)
    return xgb_r2, xgb_rmse
