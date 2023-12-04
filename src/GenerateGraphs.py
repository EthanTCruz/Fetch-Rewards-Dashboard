import pandas as pd
import plotly.graph_objects as go



class Grapher():
    def __init__(self,dataFileLocation: str) -> None:
        self.dataFileLocation = dataFileLocation

    def returnMonthlyData(self):
        data_csv = pd.read_csv(self.dataFileLocation)

        data_csv['Date'] = pd.to_datetime(data_csv['# Date'])
        data_csv.drop(columns=['# Date'], inplace=True)

        data_monthly = data_csv.groupby(pd.Grouper(key='Date', freq='M')).sum()
        return data_monthly

    def Graph(self,predicted_data, data_monthly, predicted_months: pd.DatetimeIndex, graph_name: str):
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

    def GenerateGraphs(self,predicted_data,graph_name):
        data_monthly = self.returnMonthlyData()
        predicted_months = pd.date_range(start=data_monthly.index[-1], periods=13, freq='M')

        new_data = pd.DataFrame({
                        'Predicted_Receipts': data_monthly.values[-1],
                        'Date': [pd.to_datetime(data_monthly.index[-1])]
                                })
        predicted_data = pd.concat([new_data, predicted_data])
        graph = self.Graph(predicted_data=predicted_data,predicted_months=predicted_months,data_monthly=data_monthly,graph_name=graph_name)

        return graph
