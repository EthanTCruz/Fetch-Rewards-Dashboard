from arima_model import ArimaPredictor
from prophet_model import ProphetPredictor
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import json
import plotly


def main():
    ProphetGraphs()
    ArimaGraphs()
    return 0


def ArimaGraphs():
    arima = ArimaPredictor(dataFileLocation='data\data_daily.csv')
    data_monthly, predicted_data = arima.PredictNMonths()

    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')

    new_data = pd.DataFrame(data_monthly.values[-1], index=[pd.to_datetime(data_monthly.index[-1])],columns=['Predicted_Receipts'])
    predicted_data = pd.concat([new_data, predicted_data])

    print(predicted_data)
    return GraphData(predicted_data=predicted_data,predicted_months=predicted_months,data_monthly=data_monthly,graph_name='arima.png')


def GraphData2(predicted_data,data_monthly,predicted_months: pd.DatetimeIndex,graph_name: str ):
    plt.figure(figsize=(12, 6))
    plt.plot(data_monthly.index, data_monthly['Receipt_Count'], marker='o')
    plt.plot(predicted_months,predicted_data, marker='x', linestyle='--', color='red', label='Predicted Data')
    plt.title('Monthly Total Scanned Receipts for 2021')
    plt.xlabel('Month')
    plt.ylabel('Total Receipts')
    plt.grid(True)
    filePath = f"src/static/images/{graph_name}"
    plt.savefig(filePath)
    return (f"/static/images/{graph_name}")

def GraphData(predicted_data, data_monthly, predicted_months: pd.DatetimeIndex, graph_name: str):
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
    width=800,  # Set the width
    height=600  # Set the height
)
    filePathHtml = f"src/static/images/{graph_name}.html"
    fig.write_html(filePathHtml)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def ProphetGraphs():
    prophet = ProphetPredictor(dataFileLocation='data\data_daily.csv')
    data_monthly = prophet.returnMonthlyData()
    predicted_data = prophet.PredictNDays()
    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')
    new_data = pd.Series([data_monthly['Receipt_Count'][-1]], index=[pd.to_datetime(data_monthly.index[-1])])
    predicted_data = pd.concat([new_data, predicted_data])
    print(predicted_data)
        # Convert the Series to a DataFrame
    predicted_data_df = predicted_data.to_frame()

    # Rename the column
    predicted_data_df.columns = ['Predicted_Receipts']


    return GraphData(predicted_data=predicted_data_df,predicted_months=predicted_months,data_monthly=data_monthly,graph_name='prophet.png')



if __name__ == "__main__":
    main()