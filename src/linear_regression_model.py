import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

class LinearRegressionPredictor:
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

        n = len(data)
        p = 2  # number of parameters (intercept and slope)
        dof = n - p  # degrees of freedom

        # Calculate standard error
        y_pred = beta_zero + beta_one * data['Days_Since_Start']
        residual = data['Receipt_Count'] - y_pred
        residual_sum_of_squares = sum(residual**2)
        sigma_squared = residual_sum_of_squares / dof
        se_beta_zero = (sigma_squared * (1/n + x_mean**2 / ss_xx)) ** 0.5
        se_beta_one = (sigma_squared / ss_xx) ** 0.5
        t_statistic = stats.t.ppf(0.975, dof)  # two-tailed test

        s = (sum((y - y_mean)**2 for y in data['Receipt_Count'])/(dof)) ** 0.5
        seb1 = (s/(sum((x - x_mean)**2 for x in data['Days_Since_Start'])))

        ci_beta_zero = (beta_zero - t_statistic * se_beta_zero, beta_zero + t_statistic * se_beta_zero)
        ci_beta_one = (beta_one - t_statistic * se_beta_one, beta_one + t_statistic * se_beta_one)



        coefficients = {
            "b0": beta_zero,
            "b1": beta_one,
            "ci_b0": ci_beta_zero,
            "ci_b1": ci_beta_one
        }
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
        ci_bounds = predicted_df
        ci_bounds["lower Receipt_Count"] = (coefficients["ci_b0"][0] + ci_bounds['Days_From_Start'] * coefficients["ci_b1"][0])
        ci_bounds["upper Receipt_Count"] = (coefficients["ci_b0"][1] + ci_bounds['Days_From_Start'] * coefficients["ci_b1"][1])
        predicted_df.drop(columns=['Days_From_Start'], inplace=True)
        results = {"Predicted_Receipts":predicted_df,
                   "conf_int":ci_bounds }
        return(results)


    
    def predict_by_months(self,year: int = 2022):
        temp = self.predict_year(year=year)
        predicted_df = temp["Predicted_Receipts"]



        predicted_df['Month'] = predicted_df['Date'].dt.to_period('M').dt.to_timestamp('M')
        monthly_sum = predicted_df.groupby('Month')['Predicted_Receipts'].sum().reset_index()
        

        ci_bounds = temp["conf_int"]
        ci_bounds['Month'] = ci_bounds['Date'].dt.to_period('M').dt.to_timestamp('M')
        upper_monthly_sum = ci_bounds.groupby('Month')['upper Receipt_Count'].sum().reset_index()
        lower_monthly_sum = ci_bounds.groupby('Month')['lower Receipt_Count'].sum().reset_index()
        ci_bounds = pd.merge(lower_monthly_sum, upper_monthly_sum, on='Month', suffixes=('_lower', '_upper'))

        monthly_sum['Date'] = monthly_sum['Month']
        monthly_sum.drop(columns=['Month'], inplace=True)
        monthly_sum.set_index('Date', inplace=True)
        
        df_reset = monthly_sum.reset_index()
        df_reset.columns = ['Date', 'Predicted_Receipts']
        df_reset = df_reset[['Predicted_Receipts', 'Date']]

        results = {"monthly_sum":df_reset,
                   "conf_int":ci_bounds
                   }
        # results = {"monthly_sum":monthly_sum,
        #            "mean":predicted_mean,
        #            "conf_int":predicted_conf_int}
        return(results)
    