from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd


class ArimaPredictor:
    def __init__(self,dataFileLocation: str) -> None:
        self.dataFileLocation = dataFileLocation

    def returnMonthlyData(self):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        data_monthly = data_csv.groupby(pd.Grouper(key='Date', freq='M')).sum()
        return data_monthly
    
    def PredictNMonths(self,numberOfMonths: int = 12,start_date: str = '2022-01-01',p: int = 1,q: int = 1,d: int = 1):
        data_monthly = self.returnMonthlyData()
        model = ARIMA(data_monthly['Receipt_Count'], order=(p, d, q))



        # Fit the model
        arima_result = model.fit()

        # Making predictions for 2022 (12 months)
        predictions_2022 = arima_result.forecast(steps=numberOfMonths)

        # Creating a DataFrame for the predictions to display them neatly
        predictions_2022_df = pd.DataFrame({'Month': pd.date_range(start=start_date, periods=numberOfMonths, freq='M'),
                                            'Predicted_Receipts': predictions_2022})

        predictions_2022_df.set_index('Month', inplace=True)
        return (data_monthly,predictions_2022_df.round())  # rounding off to nearest whole number for readability
