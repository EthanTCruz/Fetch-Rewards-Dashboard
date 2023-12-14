from Fetch_Dash.classes.FlaskDash import ReceiptsDashboard
from waitress import serve

def run():
    app = ReceiptsDashboard()
    app.Run()

def get_flask_server():
    app = ReceiptsDashboard()
    app = app.get_server()
    return app
    
if __name__ == "__main__":
    app = ReceiptsDashboard()
    flask_app = app.get_server()
    serve(flask_app, host='0.0.0.0', port=8050)
