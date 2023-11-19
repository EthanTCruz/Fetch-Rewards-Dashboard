
from flask import Flask, render_template, request, make_response
from main import ArimaGraphs, ProphetGraphs

app = Flask(__name__)

@app.route('/')
@app.route('/toggle')
def dashboard():
    # Default plot type
    plot_type = 'arima'
    # Check if a session cookie exists for plot type
    if 'plot_type' in request.cookies:
        # Toggle the plot type
        plot_type = 'prophet' if request.cookies.get('plot_type') == 'arima' else 'arima'
    
    # Prepare button text based on the plot type
    button_text = "Show Prophet Plot" if plot_type == 'arima' else "Show ARIMA Plot"
    
    # Generate image paths
    arima_image = ArimaGraphs()
    prophet_image = ProphetGraphs()

    # Create response with the new plot type
    response = make_response(render_template('dashboard.html', plot_type=plot_type, button_text=button_text, arima_image=arima_image, prophet_image=prophet_image))
    response.set_cookie('plot_type', plot_type)
    return response

if __name__ == '__main__':
    app.run(debug=True)