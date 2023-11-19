from arima_model import ArimaPredictor
from prophet_model import ProphetPredictor
import matplotlib.pyplot as plt
import pandas as pd

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


def GraphData(predicted_data,data_monthly,predicted_months: pd.DatetimeIndex,graph_name: str ):
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

def ProphetGraphs():
    prophet = ProphetPredictor(dataFileLocation='data\data_daily.csv')
    data_monthly = prophet.returnMonthlyData()
    predicted_data = prophet.PredictNDays()
    predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')
    new_data = pd.Series([data_monthly['Receipt_Count'][-1]], index=[pd.to_datetime(data_monthly.index[-1])])
    predicted_data = pd.concat([new_data, predicted_data])
    print(predicted_data)
    return GraphData(predicted_data=predicted_data,predicted_months=predicted_months,data_monthly=data_monthly,graph_name='prophet.png')



if __name__ == "__main__":
    main()