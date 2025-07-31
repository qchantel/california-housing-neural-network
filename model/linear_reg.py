from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def create_linear_reg_model():
    model = LinearRegression()
    return model

def train_linear_reg_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_linear_reg_model(model, X_test, y_test):
    lr_predictions = model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_predictions)
    return lr_r2

def linear_reg_r2_score(X_train, y_train, X_test, y_test):
    model = create_linear_reg_model()
    model = train_linear_reg_model(model, X_train, y_train)
    lr_r2 = evaluate_linear_reg_model(model, X_test, y_test)
    return lr_r2
