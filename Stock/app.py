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
        number_of_days = int(request.args.get('days'))

        # Download historical stock data (e.g., for the last 6 months)
        df = yf.download(tickers=ticker_value, period='6mo', interval='1d')
        if df.empty:
            return render_template('error.html', message="No historical data available for the given ticker.")
    except:
        return render_template('error.html', message="Invalid input or unable to fetch data.")

    if number_of_days < 1 or number_of_days > 365:
        return render_template('error.html', message="Number of days must be between 1 and 365.")

    # Plot historical candlestick chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Historical Data'
    ))
    fig.update_layout(
        title=f'{ticker_value} Historical Candlestick Chart (6 Months)',
        yaxis_title='Stock Price (in USD)',
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    plot_div = plot(fig, auto_open=False, output_type='div')

    # Prepare data for machine learning prediction
    df_ml = df[['Close']]
    df_ml['Prediction'] = df_ml[['Close']].shift(-number_of_days)

    X = np.array(df_ml.drop(['Prediction'], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-number_of_days:]
    X = X[:-number_of_days]

    y = np.array(df_ml['Prediction'])
    y = y[:-number_of_days]

    # Train-test split and model training
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    # Make predictions
    forecast_prediction = clf.predict(X_forecast)

    # Prepare predictions for plotting
    pred_dict = {"Date": [], "Prediction": []}
    for i in range(len(forecast_prediction)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast_prediction[i])
    pred_df = pd.DataFrame(pred_dict)

    # Plot prediction graph
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'], name='Prediction')])
    pred_fig.update_layout(
        title=f'{ticker_value} Stock Price Prediction',
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    return render_template(
        'result.html',
        plot_div=plot_div,
        confidence=confidence,
        forecast=forecast_prediction,
        plot_div_pred=plot_div_pred,
        ticker=ticker_value,
        pred_df=pred_df
    )

if __name__ == '__main__':
    app.run(debug=True)
