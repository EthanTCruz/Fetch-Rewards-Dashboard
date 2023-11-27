from arima_model import ArimaPredictor
from prophet_model import ProphetPredictor
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import json
import plotly


def main():
#    ProphetGraphs()
#    ArimaGraphs()
    return 0



def ArimaGraphs():
    arima = ArimaPredictor(dataFileLocation='data\data_daily.csv')
    data_monthly, predicted_data = arima.PredictNMonths()

    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')

    new_data = pd.DataFrame(data_monthly.values[-1], index=[pd.to_datetime(data_monthly.index[-1])],columns=['Predicted_Receipts'])
    predicted_data = pd.concat([new_data, predicted_data])


    return Graph(predicted_data=predicted_data,predicted_months=predicted_months,data_monthly=data_monthly,graph_name='arima.png')


def Graph(predicted_data, data_monthly, predicted_months: pd.DatetimeIndex, graph_name: str):
    summary_statistics = {}
    summary_statistics["Projected 2022 Mean"] = round(predicted_data["Predicted_Receipts"].mean(),2)
    summary_statistics["Projected 2022 Min"] = int(round(predicted_data["Predicted_Receipts"].min(),0))
    summary_statistics["Projected 2022 Max"] = int(round(predicted_data["Predicted_Receipts"].max(),0))
    summary_statistics["Projected 2022 Sum"] = int(round(predicted_data["Predicted_Receipts"].sum(),0))

    summary_statistics["2021 Mean"] = round(data_monthly["Receipt_Count"].mean(),2)
    summary_statistics["2021 Min"] = int(round(data_monthly["Receipt_Count"].min(),0))
    summary_statistics["2021 Max"] = int(round(data_monthly["Receipt_Count"].max(),0))
    summary_statistics["2021 Sum"] = int(round(data_monthly["Receipt_Count"].sum(),0))

    # Create Plotly figure
    fig = go.Figure()

    # Add predicted data
    fig.add_trace(go.Scatter(x=predicted_months, y=predicted_data['Predicted_Receipts'].values, 
                             mode='lines+markers', name='Projected 2022 Data', 
                             line=dict(color='red', dash='dash')))
    # Add actual data
    fig.add_trace(go.Scatter(x=data_monthly.index, y=data_monthly['Receipt_Count'].values, 
                             mode='lines+markers', name='2021 Data',
                             line=dict(color='blue', dash='solid')))



    # Update layout
    fig.update_layout(title='Projected 2022 Monthly Scans',
                      xaxis_title='Month',
                      yaxis_title='Receipts Scanned (per month)',
                      template='plotly_dark')
    fig.update_layout(
        autosize=True
    )

    filePathHtml = f"src/static/images/{graph_name}.html"
    fig.write_html(filePathHtml)

    return fig, summary_statistics

def GraphData(predicted_data, data_monthly, predicted_months: pd.DatetimeIndex, graph_name: str):
    summary_statistics = {}
    summary_statistics[f"{graph_name[0:-4]}_predicted_mean"] = predicted_data["Predicted_Receipts"].mean()
    summary_statistics[f"{graph_name[0:-4]}_predicted_min"] = predicted_data["Predicted_Receipts"].min()
    summary_statistics[f"{graph_name[0:-4]}_predicted_max"] = predicted_data["Predicted_Receipts"].max()
    summary_statistics[f"{graph_name[0:-4]}_predicted_sum"] = predicted_data["Predicted_Receipts"].sum()

    summary_statistics["actual_mean"] = data_monthly["Receipt_Count"].mean()
    summary_statistics["actual_min"] = data_monthly["Receipt_Count"].min()
    summary_statistics["actual_max"] = data_monthly["Receipt_Count"].max()
    summary_statistics["actual_sum"] = data_monthly["Receipt_Count"].sum()

    # Create Plotly figure
    fig = go.Figure()

    # Add actual data
    fig.add_trace(go.Scatter(x=data_monthly.index, y=data_monthly['Receipt_Count'].values, 
                             mode='lines+markers', name='Actual Data'))

    # Add predicted data
    fig.add_trace(go.Scatter(x=predicted_months, y=predicted_data['Predicted_Receipts'].values, 
                             mode='lines+markers', name='Predicted Data', 
                             line=dict(color='red', dash='dash')))

    # Update layout
    fig.update_layout(title='Monthly Total Scanned Receipts for 2021',
                      xaxis_title='Month',
                      yaxis_title='Total Receipts',
                      template='plotly_dark')
    fig.update_layout(
        autosize=True
    )

    filePathHtml = f"src/static/images/{graph_name}.html"
    fig.write_html(filePathHtml)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON, summary_statistics

def ProphetGraphs():
    prophet = ProphetPredictor(dataFileLocation='data\data_daily.csv')
    data_monthly = prophet.returnMonthlyData()
    predicted_data = prophet.PredictNDays()
    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')
    new_data = pd.Series([data_monthly['Receipt_Count'][-1]], index=[pd.to_datetime(data_monthly.index[-1])])
    predicted_data = pd.concat([new_data, predicted_data])

        # Convert the Series to a DataFrame
    predicted_data_df = predicted_data.to_frame()

    # Rename the column
    predicted_data_df.columns = ['Predicted_Receipts']


    return Graph(predicted_data=predicted_data_df,predicted_months=predicted_months,data_monthly=data_monthly,graph_name='prophet.png')



if __name__ == "__main__":
    main()