import pandas as pd
import numpy as np

class LinearRegressionPredictor:
    def __init__(self, data) -> None:
        self.data = data
        self.coefficients = None
        
    def PredictNMonths(self, N=12):
        # Calculate the last day in the dataset
        last_day = self.data['# Date'].max()

        # Generate dates for the next N months
        future_months = pd.date_range(start=last_day, periods=N + 1, freq='M')[1:]

        # Calculate 'Days' for each future month
        future_days = (future_months - self.data['# Date'].min()).days

        # Predict for each future month
        future_predictions = self.predict(future_days.values.reshape(-1, 1))

        return pd.DataFrame({'Predicted_Receipts': future_predictions}, index=future_months)
    def fit(self, X, y):
        # Adding a column of ones to X for the intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal Equation (Least Squares Method)
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def trainModel(self, x_columns, y_column):
        X = self.data[x_columns].values
        y = self.data[y_column].values
        self.fit(X, y)
        return self.coefficients

    def predict(self, X):
        # Adding a column of ones to X for the intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Making predictions
        return X_b.dot(self.coefficients)

