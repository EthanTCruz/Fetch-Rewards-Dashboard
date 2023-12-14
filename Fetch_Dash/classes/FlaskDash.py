# Import required libraries
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
from Fetch_Dash.classes.GenerateGraphs import Grapher
from dash import dash_table
from dash.dependencies import Input, Output
from flask_caching import Cache
from Fetch_Dash.config.config import Settings
import cowsay
s= Settings()



class ReceiptsDashboard:

    def get_server(self):
        
        app = dash.Dash(__name__)
        # Create a Dash application
        grapher = Grapher()

        cowsay.cow("Creating Arima Model")
        grapher.ArimaGraphs(refresh_data=True)

        cowsay.cow("Creating Linear and Prophet Models")
        grapher.LinearRegressionGraphs(refresh_data=True)
        grapher.ProphetGraphs(refresh_data=True)

        cowsay.cow("Creating Simple RNN Model")
        grapher.RNNGraphs(refresh_data=True)

        cowsay.cow("Creating LSTM Model")
        grapher.LSTMGraphs(refresh_data=True)




        # Setup cache
        cache = Cache(app.server, config={
            'CACHE_TYPE': 'filesystem',
            'CACHE_DIR': f'{s.cache_dir}'
        })

        app.layout = html.Div([
                dcc.Dropdown(
                id='graph-dropdown',
                options=[
                    {'label': 'LSTM Recurrent Neural Network (Summed By Month)', 'value': 'lstm'},
                    {'label': 'ARIMA (Summed By Month)', 'value': 'ARIMA'},
                    {'label': 'Simple Recurrent Neural Network (Summed By Month)', 'value': 'RNN'},
                    {'label': 'Prophet (Summed By Month)', 'value': 'Prophet'},
                    {'label': 'Linear Regression (Summed By Month)', 'value': 'Linear Regression'}
                ],
                value='lstm'  # Default value
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
                            style_cell={'textAlign': 'center'},
                        )
                    ], style={'marginBottom': '20px'}),

                    html.Div([
                        dash_table.DataTable(
                            id='summary-stats-actual-table',
                            columns=[{'name': '2021 Actual Statistics', 'id': 'Statistic'},
                                    {'name': 'Value', 'id': 'Value'}],
                            style_cell={'textAlign': 'center'}, 
                        )
                    ])
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle'})
            ], style={'display': 'flex'})
        ])




        @cache.memoize(timeout=10)
        def generate_stats_and_figs(model_type):
            # Your existing code to generate summary statistics based on model_type
            if model_type == 'ARIMA':
                fig, summary_stats = grapher.ArimaGraphs()
            elif model_type == 'Prophet':
                fig, summary_stats = grapher.ProphetGraphs()
            elif model_type == "Linear Regression":
                fig,summary_stats = grapher.LinearRegressionGraphs()
            elif model_type == "RNN":
                fig,summary_stats = grapher.RNNGraphs()
            elif model_type == "lstm":
                fig,summary_stats = grapher.LSTMGraphs()
            else:
                return 0
            return fig, summary_stats



        @app.callback(
            Output('graph', 'figure'),
            Output('summary-stats-actual-table', 'data'),
            Output('summary-stats-predicted-table', 'data'), 
            [Input('graph-dropdown', 'value')]
        )
        def update_content(selected_model):
            fig, summary_stats = generate_stats_and_figs(model_type=selected_model)

            stats_df = pd.DataFrame(list(summary_stats.items()), columns=['Statistic', 'Value'])
            stats_df['Value'] = stats_df['Value'].apply(format_number)  # Format numbers
            df_2021 = stats_df[stats_df['Statistic'].str.startswith('2021')]
            df_predicted = stats_df[~stats_df['Statistic'].str.startswith('2021')]

            return fig, df_2021.to_dict('records'), df_predicted.to_dict('records')

        def format_number(value):
            if isinstance(value, int):
                return f"{value:,}"
            elif isinstance(value, float):
                return f"{value:,.2f}"
            else:
                return value


        return app.server

    def Run(self):
        app = self.get_server()
        app.run_server(debug=False, host=s.HOST, port=8050)




