from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd


class ArimaPredictor:
    def __init__(self,dataFileLocation: str) -> None:
        self.dataFileLocation = dataFileLocation

    def column_month_summation(self,df,column_name,start_date: str ='2022-01-01'):

        temp_df = pd.DataFrame({'Date': pd.date_range(start=start_date, periods=365, freq='D'),
                            'Predicted_Receipts': df[column_name]})
        df['Month'] = temp_df['Date'].dt.to_period('M').dt.to_timestamp('M')
        monthly_sum = df.groupby('Month')[column_name].sum().reset_index()
        monthly_sum['Date'] = monthly_sum['Month']
        monthly_sum.drop(columns=['Month'], inplace=True)
        df = monthly_sum
        return df


    def PredictNMonths(self,start_date: str = '2022-01-01',p: int = 1,q: int = 1,d: int = 1):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        model = ARIMA(data_csv['Receipt_Count'], order=(p, d, q))

        # Fit the model
        arima_result = model.fit()

        # Making predictions for 2022 (12 months)
        predictions_2022 = arima_result.forecast(steps=365)
        forecast = arima_result.get_forecast(steps=365)
        predicted_conf_int = forecast.conf_int(alpha=0.05)



        

        ci_lower_predictions_2022_df = self.column_month_summation(df=predicted_conf_int,column_name="upper Receipt_Count")


        ci_upper_predictions_2022_df = self.column_month_summation(df=predicted_conf_int,column_name="lower Receipt_Count")

        predicted_conf_int = pd.merge(ci_lower_predictions_2022_df, ci_upper_predictions_2022_df, on='Date', suffixes=('_lower', '_upper'))
        predicted_conf_int.set_index('Date', inplace=True)

        
        #forecast = predictions_2022.predicted_mean
        #conf_int = predictions_2022.conf_int(alpha=0.05)

        # Creating a DataFrame for the predictions to display them neatly
        #predictions_2022_df = pd.DataFrame({'Date': pd.date_range(start=start_date, periods=365, freq='D'),
        #                                    'Predicted_Receipts': predictions_2022})



        # predictions_2022_df['Month'] = ci_upper_predictions_2022_df['Date'].dt.to_period('M').dt.to_timestamp('M')
        # monthly_sum = predictions_2022_df.groupby('Month')['Predicted_Receipts'].sum().reset_index()
        # monthly_sum['Date'] = monthly_sum['Month']
        # monthly_sum.drop(columns=['Month'], inplace=True)

        df = predictions_2022.to_frame()

        # Now, rename the column
        df.columns = ['Predicted_Receipts']
        monthly_sum = self.column_month_summation(df=df,column_name='Predicted_Receipts')
        results = {"monthly_sum":monthly_sum,
                   "conf_int":predicted_conf_int}
        return (results)  # rounding off to nearest whole number for readability
