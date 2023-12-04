from arima_model import ArimaPredictor
from prophet_model import ProphetPredictor
from config import Settings
import pandas as pd
import plotly.graph_objects as go

from linear_regression_model import LinearProgressionPredictor

s = Settings()
dataFileLocation = s.dataFileLocation

def main():
    ProphetGraphs()
    lrm = LinearProgressionPredictor(dataFileLocation='data\data_daily.csv')
    predicted_data = lrm.predict_by_months()
    t = Graph2(predicted_data=predicted_data,graph_name='lrm.png')
    print(LinearRegressionGraphs() == t)

    arima = ArimaPredictor(dataFileLocation='data\data_daily.csv')
    predicted_data = arima.PredictNMonths()
    a = Graph2(predicted_data=predicted_data,graph_name='arima.png')
    print(ArimaGraphs() == a)

    prophet = ProphetPredictor(dataFileLocation=dataFileLocation)
    predicted_data = prophet.PredictNDays2()
    p = Graph2(predicted_data=predicted_data,graph_name='prophet.png')
    print(p == ProphetGraphs())
    #lr = LinearProgressionPredictor(dataFileLocation=dataFileLocation)
    #print(lr.generate_coefficients())
    #data = modelPrediector().predict
    # Graph(name = model) 
    return 0



def returnMonthlyData():
    data_csv = pd.read_csv(dataFileLocation)

    data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
    data_csv.drop(columns=['# Date'], inplace=True)

    data_monthly = data_csv.groupby(pd.Grouper(key='Date', freq='M')).sum()
    return data_monthly

def Graph2(predicted_data,graph_name):
    data_monthly = returnMonthlyData()
    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')

    new_data = pd.DataFrame(data_monthly.values[-1], index=[pd.to_datetime(data_monthly.index[-1])],columns=['Predicted_Receipts'])
    predicted_data = pd.concat([new_data, predicted_data])


    return Graph(predicted_data=predicted_data,predicted_months=predicted_months,data_monthly=data_monthly,graph_name=graph_name)

def LinearRegressionGraphs():
    lrm = LinearProgressionPredictor(dataFileLocation='data\data_daily.csv')
    predicted_data = lrm.predict_by_months()
    data_monthly = returnMonthlyData()
    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')
    new_data = pd.DataFrame(data_monthly.values[-1], index=[pd.to_datetime(data_monthly.index[-1])],columns=['Predicted_Receipts'])
    predicted_data = pd.concat([new_data, predicted_data])
    return Graph(predicted_data=predicted_data,predicted_months=predicted_months,data_monthly=data_monthly,graph_name='lrm.png')

def LinearRegressionGraphs2():
    lrm = LinearProgressionPredictor(dataFileLocation='data\data_daily.csv')
    predicted_data = lrm.predict_by_months2()
    data_monthly = returnMonthlyData()
    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')

    new_data = pd.DataFrame(data_monthly.values[-1], index=[pd.to_datetime(data_monthly.index[-1])],columns=['Predicted_Receipts'])
    predicted_data = pd.concat([new_data, predicted_data])


    return Graph(predicted_data=predicted_data,predicted_months=predicted_months,data_monthly=data_monthly,graph_name='lrm.png')

def ArimaGraphs():
    arima = ArimaPredictor(dataFileLocation='data\data_daily.csv')
    data_monthly = returnMonthlyData()
    predicted_data = arima.PredictNMonths()

    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')

    new_data = pd.DataFrame(data_monthly.values[-1], index=[pd.to_datetime(data_monthly.index[-1])],columns=['Predicted_Receipts'])
    predicted_data = pd.concat([new_data, predicted_data])


    return Graph(predicted_data=predicted_data,predicted_months=predicted_months,data_monthly=data_monthly,graph_name='arima.png')


def Graph(predicted_data, data_monthly, predicted_months: pd.DatetimeIndex, graph_name: str):
    summary_statistics = {}
    summary_statistics["Projected 2022 Monthly Mean"] = round(predicted_data["Predicted_Receipts"].mean(),2)
    summary_statistics["Projected 2022 Monthly Min"] = int(round(predicted_data["Predicted_Receipts"].min(),0))
    summary_statistics["Projected 2022 Monthly Max"] = int(round(predicted_data["Predicted_Receipts"].max(),0))
    summary_statistics["Projected 2022  Sum"] = int(round(predicted_data["Predicted_Receipts"].sum(),0))

    summary_statistics["2021 Monthly Mean"] = round(data_monthly["Receipt_Count"].mean(),2)
    summary_statistics["2021 Monthly Min"] = int(round(data_monthly["Receipt_Count"].min(),0))
    summary_statistics["2021 Monthly Max"] = int(round(data_monthly["Receipt_Count"].max(),0))
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


    return fig, summary_statistics


def ProphetGraphs():
    prophet = ProphetPredictor(dataFileLocation='data\data_daily.csv')
    data_monthly = prophet.returnMonthlyData()
    predicted_data = prophet.PredictNDays()
    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')
    new_data = pd.Series([data_monthly['Receipt_Count'].iloc[-1]], index=[pd.to_datetime(data_monthly.index[-1])])
    predicted_data = pd.concat([new_data, predicted_data])

        # Convert the Series to a DataFrame
    predicted_data_df = predicted_data.to_frame()

    # Rename the column
    predicted_data_df.columns = ['Predicted_Receipts']


    return Graph(predicted_data=predicted_data_df,predicted_months=predicted_months,data_monthly=data_monthly,graph_name='prophet.png')



if __name__ == "__main__":
    main()