# Import required libraries
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
from GenerateGraphs import Grapher
from dash import dash_table
from dash.dependencies import Input, Output


# Create a Dash application
app = dash.Dash(__name__)
grapher = Grapher()



app.layout = html.Div([

        dcc.Dropdown(
        id='graph-dropdown',
        options=[
            {'label': 'ARIMA', 'value': 'ARIMA'},
            {'label': 'Linear Regression', 'value': 'Linear Regression'},
            {'label': 'Prophet', 'value': 'Prophet'}
            
        ],
        value='ARIMA'  # Default value
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='graph')
        ], style={'width': '70%', 'display': 'inline-block'}),

        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='summary-stats-predicted-table',
                    columns=[{'name': '2022 Predicted Statistics', 'id': 'Statistic'},
                             {'name': 'Value', 'id': 'Value'}],
                    style_cell={'textAlign': 'center'},  # Center-align text in all cells
                    # Add any other properties and styling for your DataTable here
                )
            ], style={'margin-bottom': '20px'}),

            html.Div([
                dash_table.DataTable(
                    id='summary-stats-actual-table',
                    columns=[{'name': '2021 Actual Statistics', 'id': 'Statistic'},
                             {'name': 'Value', 'id': 'Value'}],
                    style_cell={'textAlign': 'center'},  # Center-align text in all cells
                    # Add any other properties and styling for your DataTable here
                )
            ])
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'})
    ], style={'display': 'flex'})
])







# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    Output('summary-stats-actual-table', 'data'),
    Output('summary-stats-predicted-table', 'data'), 
    [Input('graph-dropdown', 'value')]
)
def update_content(selected_model):
    if selected_model == 'ARIMA':

        fig, summary_stats = grapher.ArimaGraphs()
    elif selected_model == 'Prophet':

        fig, summary_stats = grapher.ProphetGraphs()
    elif selected_model == "Linear Regression":
        fig,summary_stats = grapher.LinearRegressionGraphs()
    else:
        # Handle other cases or default case
        return 0



    # Convert summary statistics to DataFrame
    stats_df = pd.DataFrame(list(summary_stats.items()), columns=['Statistic', 'Value'])
    stats_df['Value'] = stats_df['Value'].apply(format_number)  # Format numbers
    df_2021 = stats_df[stats_df['Statistic'].str.startswith('2021')]
    df_predicted = stats_df[~stats_df['Statistic'].str.startswith('2021')]



    return fig, df_2021.to_dict('records'), df_predicted.to_dict('records')

def format_number(value):
    if isinstance(value, int):
        # Format integers without decimal places
        return f"{value:,}"
    elif isinstance(value, float):
        # Format floats with two decimal places
        return f"{value:,.2f}"
    else:
        # Return non-numeric values as is
        return value


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
