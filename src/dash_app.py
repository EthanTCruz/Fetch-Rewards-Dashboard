# Import required libraries
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from main import ArimaGraphs, ProphetGraphs
from dash import dash_table

# Create a Dash application
app = dash.Dash(__name__)



app.layout = html.Div([
    html.Button('Toggle Graph', id='toggle-button', n_clicks=0),
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
    [Output('graph', 'figure'), 
     Output('summary-stats-actual-table', 'data'),
     Output('summary-stats-predicted-table', 'data'),   # Update this line
     Output('toggle-button', 'children')],
    [Input('toggle-button', 'n_clicks')]
)
def update_content(n_clicks):
    if n_clicks % 2 == 0:
        fig, summary_stats = ArimaGraphs()
        button_label = "Show Prophet"
    else:
        fig, summary_stats = ProphetGraphs()
        button_label = "Show ARIMA"

    # Convert summary statistics to DataFrame
    stats_df = pd.DataFrame(list(summary_stats.items()), columns=['Statistic', 'Value'])
    stats_df['Value'] = stats_df['Value'].apply(format_number)  # Format numbers
    df_2021 = stats_df[stats_df['Statistic'].str.startswith('2021')]
    df_predicted = stats_df[~stats_df['Statistic'].str.startswith('2021')]



    return fig, df_2021.to_dict('records'), df_predicted.to_dict('records'), button_label

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
