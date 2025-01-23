from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection
import plotly.graph_objects as go
from plotly.offline import plot

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        ticker_value = request.args.get('ticker').upper()
        df = yf.download(tickers=ticker_value, period='1d', interval='1m')
    except:
        return render_template('error.html', message="Invalid ticker symbol or unable to fetch data.")

    try:
        number_of_days = int(request.args.get('days'))
    except:
        return render_template('error.html', message="Invalid number of days format.")

    if number_of_days < 0 or number_of_days > 365:
        return render_template('error.html', message="Number of days must be between 1 and 365.")
    
    # Check if the ticker is an Indian stock (based on the ticker symbol)
    if ticker_value.endswith('.NS'):  # Indian stocks on NSE usually have '.NS' suffix
        currency = 'INR'
    else:
        currency = 'USD'  # Assume foreign stocks are in USD for simplicity

    # Generate stock candlestick graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                  open=df['Open'],
                                  high=df['High'],
                                  low=df['Low'],
                                  close=df['Close'],
                                  name='Market Data'))
    fig.update_layout(
        title=f'{ticker_value} Live Share Price Evolution',
        yaxis_title=f'Stock Price ({currency} per Share)',
        xaxis_rangeslider_visible=True,
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    plot_div = plot(fig, auto_open=False, output_type='div')

    # Machine learning prediction
    df_ml = yf.download(tickers=ticker_value, period='3mo', interval='1h')
    df_ml = df_ml[['Adj Close']]
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-number_of_days)

    X = np.array(df_ml.drop(['Prediction'], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-number_of_days:]
    X = X[:-number_of_days]
    y = np.array(df_ml['Prediction'])
    y = y[:-number_of_days]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    forecast_prediction = clf.predict(X_forecast)

    # Prediction plot
    pred_dict = {"Date": [], "Prediction": []}
    for i in range(len(forecast_prediction)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast_prediction[i])
    pred_df = pd.DataFrame(pred_dict)

    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'], name='Prediction')])
    pred_fig.update_layout(
        title=f'{ticker_value} Stock Price Prediction',
        xaxis_rangeslider_visible=True,
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    return render_template('result.html', plot_div=plot_div, confidence=confidence, forecast=forecast_prediction, plot_div_pred=plot_div_pred, ticker=ticker_value, currency=currency, pred_df=pred_df)



if __name__ == '__main__':
    app.run(debug=True)
