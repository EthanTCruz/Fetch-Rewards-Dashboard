import numpy as np
import pandas as pd
from datetime import datetime

class LinearProgressionPredictor:
    def __init__(self,dataFileLocation: str) -> None:
        self.dataFileLocation = dataFileLocation

    def returnDailyData(self):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        return data_csv
    
    def generate_coefficients(self):
        data = self.returnDailyData() 

        data['Date'] = pd.to_datetime(data['Date'])

        # Subtract the first date and convert to days, then add 1
        data['Days_Since_Start'] = (data['Date'] - data['Date'].iloc[0]).dt.days
        #sum_of_squared_deviations = sum((x - mean) ** 2 for x in data)

        x_mean = data['Days_Since_Start'].mean()
        y_mean = data['Receipt_Count'].mean()


        ss_xx = sum((x - x_mean)**2 for x in data['Days_Since_Start'])
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(data['Days_Since_Start'],data['Receipt_Count']))
        beta_one = ss_xy/ss_xx
        beta_zero = y_mean - beta_one*x_mean
        coefficients = {"b0":beta_zero,"b1":beta_one}
        #cost = sum((y - (beta_zero + x*beta_one))**2 for x, y in zip(data['Days_Since_Start'],data['Receipt_Count']) )
        #current cost = 17170506216661.105
        return (coefficients)
    
    def predict_year(self, year: int = 2022):
        coefficients = self.generate_coefficients()
        data = self.returnDailyData() 
        # Example original DataFrame
        data['Date'] = pd.to_datetime(data['Date'])

        # Find the start date of the original DataFrame
        start_date = data['Date'].min()

        # Specify the year for the new DataFrame
        year = 2022

        # Create a DataFrame for all days in the specified year
        date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        predicted_df = pd.DataFrame(date_range, columns=['Date'])

        # Calculate the number of days away from the start date
        predicted_df['Days_From_Start'] = (predicted_df['Date'] - start_date).dt.days
        predicted_df["Predicted_Receipts"] = (coefficients["b0"] + predicted_df['Days_From_Start'] * coefficients["b1"])
        predicted_df.drop(columns=['Days_From_Start'], inplace=True)

        return(predicted_df)


    
    def predict_by_months(self,year: int = 2022):
        predicted_df = self.predict_year(year=year)

        # Summing the predicted receipts by month
        predicted_df['Month'] = predicted_df['Date'].dt.to_period('M').dt.to_timestamp('M')
 

        monthly_sum = predicted_df.groupby('Month')['Predicted_Receipts'].sum().reset_index()
        
        monthly_sum['Date'] = monthly_sum['Month']
        monthly_sum.drop(columns=['Month'], inplace=True)
        monthly_sum.set_index('Date', inplace=True)
        
        df_reset = monthly_sum.reset_index()
        df_reset.columns = ['Date', 'Predicted_Receipts']
        df_reset = df_reset[['Predicted_Receipts', 'Date']]

        return(df_reset)
    