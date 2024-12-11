import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA



st.title('Stock Price Prediction')
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_returns(prices):
    """
    Calculate log returns of stock prices.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

def fit_garch_model(returns):
    """
    Fit a GARCH(1, 1) model to the stock returns.
    """
    model = arch_model(returns, vol='Garch', p=1, q=1)
    fitted_model = model.fit(disp='off')
    return fitted_model

def forecast_volatility(fitted_model, horizon):
    """
    Forecast volatility using the fitted GARCH model.
    """
    forecasts = fitted_model.forecast(horizon=horizon)
    vol_forecast = forecasts.variance.values[-1, :]
    return np.sqrt(vol_forecast)  # Standard deviation (volatility)

def predict_prices(prices, vol_forecast, forecast_days):
    """
    Predict future prices using ARIMA model and GARCH volatility forecast.
    """
    # Fit an ARIMA model on the prices
    arima_model = ARIMA(prices, order=(5, 1, 0))  # Order can be adjusted based on ACF and PACF plots
    arima_fit = arima_model.fit()

    # Forecast prices using ARIMA
    arima_forecast = arima_fit.forecast(steps=forecast_days)

    # Incorporate volatility into price forecasts (basic adjustment)
    price_forecast = arima_forecast * (1 + vol_forecast.mean())  # Adjust based on predicted volatility

    return price_forecast


def main(ticker, start_date, end_date, forecast_days):
    # Fetch and prepare data
    prices = fetch_stock_data(ticker, start_date, end_date)
    returns = calculate_returns(prices)
    st.write(prices)
    st.write(returns)

    # Fit GARCH model and forecast volatility
    garch_model = fit_garch_model(returns)
    vol_forecast = forecast_volatility(garch_model, forecast_days)
    st.write(garch_model)
    st.write(vol_forecast)

    # Predict future prices
    price_forecast = predict_prices(prices, vol_forecast, forecast_days)
    st.write(price_forecast)


if __name__ == "__main__":
    # Example usage
    ticker = st.text_input('Enter Stock')  # Ticker for Reliance Industries on NSE
    start_date = '2020-01-01'
    end_date = '2024-08-30'
    forecast_days = 10  # Predict the next 10 days

    main(ticker, start_date, end_date, forecast_days)