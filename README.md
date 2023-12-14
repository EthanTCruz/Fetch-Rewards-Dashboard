# Fetch-Rewards-Dashboard
Hello Fetch Rewards!!!

For my Dashboard I'll admit I got a little carried away, I don't often get a good enough reason to experiment with models in python.
And well, by the end I just couldn't pick one favorite so I've left you with a decently flexible Flask application showing all 5 different methods I tried.

For ranking each model by credibility I would reccommend:
1. LSTM RNN - Commonly makes most sense for sequential time series data
2. ARIMA - Statistical Model various sources recommended
3. Simple RNN - Basic implementation before I attempted LSTM, but still worthwhile to includde
4. Prophet - Similar to Linear Regression, but with a holiday component (https://facebook.github.io/prophet/)
5. Linear Refression - Most basic answer

Instructions:
To run with python from console:
`pip install -r requirements.txt`
`python setup.py build`
`python setup.py install`
`fetch_dashboard`

To run with Docker:
`docker pull ethancruz/fetch_rewards_dashboard:final`
`docker run -p 8050:8050  ethancruz/fetch_rewards_dashboard:final`


After using either of the methods for starting the app,the dashboard should now be available on 'http://localhost:8050/' as soon as the predictions from the lstm model have been computed, as will be visible in the console.


If there are any issues the code is available on the following github repo:
`https://github.com/EthanTCruz/Fetch-Rewards-Dashboard/tree/prod`

The code used in development is available here:
`https://github.com/EthanTCruz/Fetch-Rewards-Dashboard/tree/main`

And finally the docker hub link can be found here:
`https://hub.docker.com/repository/docker/ethancruz/fetch_rewards_dashboard/general`



