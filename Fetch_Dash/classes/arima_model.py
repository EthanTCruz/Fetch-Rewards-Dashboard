import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config.config import Settings



s = Settings()


class ArimaPredictor:
    def __init__(self,dataFileLocation: str = s.dataFileLocation) -> None:
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


    def calc_arima_parameters(self):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        model = pm.auto_arima(data_csv['Receipt_Count'], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=3,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=True,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=False,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

        p, d, q = model.order
        if model.seasonal_order:
            P, D, Q, m = model.seasonal_order
        else:
            P, D, Q, m = 0,0,0,0
        s.arima_order = p,d,Q
        s.arima_seasonal_order = P, D, Q, m

    def PredictNMonths(self,start_date: str = '2022-01-01',p: int = 2,q: int = 0,d: int = 2):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        if s.arima_order == s.arima_default or s.arima_seasonal_order == s.arima_default:
            self.calc_arima_parameters()

        model = SARIMAX(data_csv['Receipt_Count'], order=(s.arima_order),seasonal_order=(s.arima_seasonal_order))
        arima_result = model.fit(disp=False)

        predictions_2022 = arima_result.forecast(steps=365)
        forecast = arima_result.get_forecast(steps=365)
        predicted_conf_int = forecast.conf_int(alpha=0.05)

        ci_lower_predictions_2022_df = self.column_month_summation(df=predicted_conf_int,column_name="upper Receipt_Count")
        ci_upper_predictions_2022_df = self.column_month_summation(df=predicted_conf_int,column_name="lower Receipt_Count")

        predicted_conf_int = pd.merge(ci_lower_predictions_2022_df, ci_upper_predictions_2022_df, on='Date', suffixes=('_lower', '_upper'))
        predicted_conf_int.set_index('Date', inplace=True)

        df = predictions_2022.to_frame()
        df.columns = ['Predicted_Receipts']
        monthly_sum = self.column_month_summation(df=df,column_name='Predicted_Receipts')

        results = {"monthly_sum":monthly_sum,
                   "conf_int":predicted_conf_int}
        
        return (results)  # rounding off to nearest whole number for readability
