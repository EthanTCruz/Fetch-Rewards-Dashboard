from prophet import Prophet
import pandas as pd


class ProphetPredictor:
    def __init__(self,dataFileLocation: str) -> None:
        self.dataFileLocation = dataFileLocation

    def returnMonthlyData(self):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        data_monthly = data_csv.groupby(pd.Grouper(key='Date', freq='M')).sum()
        return data_monthly
    
    def PredictNDays(self,numberOfDays: int = 365):
        data_csv = pd.read_csv(self.dataFileLocation)

        # Preparing the data for Prophet
        data_prophet = data_csv.rename(columns={'# Date': 'ds', 'Receipt_Count': 'y'})


        prophet_model = Prophet()
        prophet_model.fit(data_prophet)
        # Create a future dataframe for the year 2022
        future = prophet_model.make_future_dataframe(periods=numberOfDays)  # Adding 365 days for the year 2022

        # Predict
        forecast = prophet_model.predict(future)

        forecast['ds'] = pd.to_datetime(forecast['ds'])
        forecast.set_index('ds', inplace=True)

        # Resample and sum to get monthly totals
        monthly_forecast = forecast['yhat'].resample('M').sum()

        # Display the monthly forecast
        return monthly_forecast[12::]


