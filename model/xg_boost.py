import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

def create_xgboost_model(n_estimators=300, learning_rate=0.1, max_depth=6):
    """Create an XGBoost regression model with default parameters."""
    model = xgb.XGBRegressor(
        random_state=42,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
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
    model = create_xgboost_model(n_estimators=300, learning_rate=0.1, max_depth=6)
    model = train_xgboost_model(model, X_train, y_train)
    xgb_r2 = evaluate_xgboost_model(model, X_test, y_test)
    xgb_rmse = rmse_xgboost_model(model, X_test, y_test)

    # # Grid search 
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 6, 9]
    # }
    # grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    # grid_search.fit(X_train, y_train)
    # best_params = grid_search.best_params_
    # print(f"Best parameters: {best_params}")
    return xgb_r2, xgb_rmse
