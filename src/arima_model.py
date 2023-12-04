from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd


class ArimaPredictor:
    def __init__(self,dataFileLocation: str) -> None:
        self.dataFileLocation = dataFileLocation

    def PredictNMonths(self,start_date: str = '2022-01-01',p: int = 1,q: int = 1,d: int = 1):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        model = ARIMA(data_csv['Receipt_Count'], order=(p, d, q))

        # Fit the model
        arima_result = model.fit()

        # Making predictions for 2022 (12 months)
        predictions_2022 = arima_result.forecast(steps=365)

        # Creating a DataFrame for the predictions to display them neatly
        predictions_2022_df = pd.DataFrame({'Date': pd.date_range(start=start_date, periods=365, freq='D'),
                                            'Predicted_Receipts': predictions_2022})




        predictions_2022_df['Month'] = predictions_2022_df['Date'].dt.to_period('M').dt.to_timestamp('M')
        monthly_sum = predictions_2022_df.groupby('Month')['Predicted_Receipts'].sum().reset_index()
        monthly_sum['Date'] = monthly_sum['Month']
        monthly_sum.drop(columns=['Month'], inplace=True)
        return (monthly_sum)  # rounding off to nearest whole number for readability
