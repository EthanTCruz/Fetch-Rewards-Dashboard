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
        data_prophet = data_csv.rename(columns={'# Date': 'ds', 'Receipt_Count': 'y'})


        prophet_model = Prophet()
        prophet_model.fit(data_prophet)
        future = prophet_model.make_future_dataframe(periods=numberOfDays)

        forecast = prophet_model.predict(future)
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        forecast.set_index('ds', inplace=True)

        monthly_forecast = forecast['yhat'].resample('M').sum()[12::]
        monthly_lower = forecast['yhat_lower'].resample('M').sum()[12::]
        monthly_upper = forecast['yhat_upper'].resample('M').sum()[12::]
        

        monthly_lower = monthly_lower.to_frame()
        monthly_lower.columns = ['lower Receipt_Count']
        monthly_lower = monthly_lower.reset_index().rename(columns={'ds': 'Date'})

        monthly_upper = monthly_upper.to_frame()
        monthly_upper.columns = ['upper Receipt_Count']
        monthly_upper = monthly_upper.reset_index().rename(columns={'ds': 'Date'})

        ci_bounds = pd.merge(monthly_lower, monthly_upper, on='Date', suffixes=('_lower', '_upper'))
        ci_bounds['Date'] = pd.to_datetime(ci_bounds['Date'])
        ci_bounds.set_index('Date', inplace=True)

        df = monthly_forecast.reset_index()
        df.columns = ['Date', 'Predicted_Receipts']
        df['Date'] = pd.to_datetime(df['Date'])

        results = {"monthly_sum":df,
                   "conf_int":ci_bounds}
        return results

