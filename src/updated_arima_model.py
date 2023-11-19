
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

class ArimaPredictor:
    def __init__(self, dataFileLocation: str) -> None:
        self.dataFileLocation = dataFileLocation

    def returnMonthlyData(self):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        data_monthly = data_csv.groupby(pd.Grouper(key='Date', freq='M')).sum()
        return data_monthly
    
    def PredictNMonths(self, numberOfMonths: int = 12, start_date: str = '2022-01-01', p: int = 1, q: int = 1, d: int = 1):
        data_monthly = self.returnMonthlyData()
        # Differencing the data
        data_monthly_diff = data_monthly.diff().dropna()

        # Fit the ARIMA model on the differenced data
        model = ARIMA(data_monthly_diff['Receipt_Count'], order=(p, d, q))
        arima_result = model.fit()

        # Making predictions on the differenced data
        predictions_diff = arima_result.forecast(steps=numberOfMonths)

        # Re-integrating the predictions to original scale
        last_data_point = data_monthly['Receipt_Count'].iloc[-1]
        predictions_cumsum = predictions_diff.cumsum()
        predictions = last_data_point + predictions_cumsum

        # Creating a DataFrame for the predictions to display them neatly
        predictions_df = pd.DataFrame({'Month': pd.date_range(start=start_date, periods=numberOfMonths, freq='M'),
                                       'Predicted_Receipts': predictions.round()})

        predictions_df.set_index('Month', inplace=True)
        return (data_monthly, predictions_df)  # Returning the original and predicted data
