from Fetch_Dash.classes.FlaskDash import ReceiptsDashboard
from waitress import serve


def get_flask_server():
    app = ReceiptsDashboard()
    return app.get_server()

flask_app = get_flask_server()


def run():
    serve(flask_app,host='0.0.0.0', port=8050)