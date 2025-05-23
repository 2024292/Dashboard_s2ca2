import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import altair as alt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


@st.cache_data
def fetch_stock_info(symbol):
    stock = yf.Ticker(symbol)
    
    return stock.info

@st.cache_data
def fetch_quarterly_financials(symbol):
    stock = yf.Ticker(symbol)
    return stock.quarterly_financials.T

@st.cache_data
def fetch_annual_financials(symbol):
    stock = yf.Ticker(symbol)
    return stock.financials.T

@st.cache_data
def fetch_weekly_price_history(symbol, period='1y'):
    """
    Fetch weekly price history for a given symbol and period.
    period: e.g. '1y', '6mo', '3mo', '1mo'
    """
    stock = yf.Ticker(symbol)
    return stock.history(period=period, interval='1wk')

@st.cache_data
def fetch_price_history_date(symbol, start_date, end_date):
    """Fetch weekly price history for a given symbol over a custom date range."""
    stock = yf.Ticker(symbol)
    # yfinance expects dates as strings or datetime objects
    return stock.history(start=start_date, end=end_date, interval='1wk')

st.title("Stock Dashboard")
symbol = st.text_input("Enter stock symbol (e.g., AAPL, TSLA):", "AAPL")

# Custom date range selection for price history
start_date, end_date = st.date_input(
    "Select start and end date for price history:",
    value=(date.today() - timedelta(days=365), date.today()),
    min_value=date(2000,1,1),
    max_value=date.today()
)

information = fetch_stock_info(symbol)

st.header('Company Information')
st.subheader(f'Ticker: {information["longName"]}')
st.subheader(f'Market Cap: {information["marketCap"]}')
st.subheader(f'Sector: {information["sector"]}')
st.subheader(f'Last Close Price: {information["previousClose"]}')
#st.subheader(f'Last Close Date: {information["lastCloseDate"]}')


# Fetch using custom date range
price_history = fetch_price_history_date(symbol, start_date, end_date)

price_history = price_history.rename_axis('Date').reset_index()
candle_stick_chart = go.Figure(data=[go.Candlestick(x=price_history['Date'],
    open=price_history['Open'],
    high=price_history['High'],
    low=price_history['Low'],
    close=price_history['Close'])])

candle_stick_chart.update_layout(xaxis_rangeslider_visible=False,)
st.plotly_chart(candle_stick_chart, use_container_width=True)


quarterly_financials = fetch_quarterly_financials(symbol)
annual_financials = fetch_annual_financials(symbol)

st.header('Financials')

selection = st.segmented_control(label = 'Period', options = ['Quarterly', 'Annual'], default = 'Quarterly')
if selection == 'Quarterly':
    quarterly_financials = quarterly_financials.rename_axis('Quarter').reset_index()
    quarterly_financials['Quarter'] = quarterly_financials['Quarter'].astype(str)
    # Ensure correct column names
    if 'Total Revenue' not in quarterly_financials.columns:
        if 'TotalRevenue' in quarterly_financials.columns:
            quarterly_financials['Total Revenue'] = quarterly_financials['TotalRevenue']
    if 'Net Income' not in quarterly_financials.columns:
        if 'NetIncome' in quarterly_financials.columns:
            quarterly_financials['Net Income'] = quarterly_financials['NetIncome']
    revenue_chart = alt.Chart(quarterly_financials).mark_bar(color='red').encode(
        x='Quarter:O',
        y='Total Revenue'
    )
    net_income_chart = alt.Chart(quarterly_financials).mark_bar(color='orange').encode(
        x='Quarter:O',
        y='Net Income'
    )
    st.altair_chart(revenue_chart, use_container_width=True)
    st.altair_chart(net_income_chart, use_container_width=True)

if selection == 'Annual':
    annual_financials = annual_financials.rename_axis('Year').reset_index()
    annual_financials['Year'] = annual_financials['Year'].astype(str).str.split('-').str[0]
    # Ensure correct column names
    if 'Total Revenue' not in annual_financials.columns:
        if 'TotalRevenue' in annual_financials.columns:
            annual_financials['Total Revenue'] = annual_financials['TotalRevenue']
    if 'Net Income' not in annual_financials.columns:
        if 'NetIncome' in annual_financials.columns:
            annual_financials['Net Income'] = annual_financials['NetIncome']
    revenue_chart = alt.Chart(annual_financials).mark_bar(color='red').encode(
        x='Year:O',
        y='Total Revenue'
    )
    net_income_chart = alt.Chart(annual_financials).mark_bar(color='orange').encode(
        x='Year:O',
        y='Net Income'
    )
    st.altair_chart(revenue_chart, use_container_width=True)
    st.altair_chart(net_income_chart, use_container_width=True)

# Forecast future prices
st.header('Price Forecast')
model_option = st.selectbox('Select forecasting model:', ['SARIMAX', 'LSTM'])
horizon = st.selectbox('Select forecast horizon (days):', [1, 3, 7], index=0)
if st.button('Run Forecast'):
    # Prepare historical close price series
    hist_df = price_history.set_index('Date')
    close_series = hist_df['Close']
    last_date = hist_df.index[-1]
    if model_option == 'SARIMAX':
        # Fit SARIMAX model
        model = SARIMAX(close_series, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
        # Forecast next periods
        forecast = result.get_forecast(steps=horizon).predicted_mean
        # Forecast dates
        fc_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
        fc_df = pd.DataFrame({'Date': fc_dates, 'Forecast': forecast.values})
        # Plot SARIMAX forecast
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=hist_df.index, y=close_series, name='Historical Close'))
        fig_fc.add_trace(go.Scatter(x=fc_df['Date'], y=fc_df['Forecast'], name='SARIMAX Forecast', line=dict(dash='dash')))
        fig_fc.update_layout(title='Historical and SARIMAX Forecasted Close Prices', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        # LSTM forecasting
        # Scale data
        data = close_series.values.reshape(-1,1)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        # Prepare sequences
        window = 5
        X, y = [], []
        for i in range(window, len(data_scaled)):
            X.append(data_scaled[i-window:i, 0])
            y.append(data_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        # Build LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(16, input_shape=(X.shape[1], 1)))
        lstm_model.dropout(0.2)
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(X, y, epochs=5, batch_size=16, verbose=0)
        # Forecast future values
        temp_input = data_scaled[-window:].tolist()
        lstm_output = []
        for _ in range(horizon):
            x_input = np.array(temp_input[-window:]).reshape(1, window, 1)
            yhat = lstm_model.predict(x_input, verbose=0)
            lstm_output.append(yhat[0, 0])
            temp_input.append(yhat[0, 0])
        # Inverse scale
        lstm_forecast = scaler.inverse_transform(np.array(lstm_output).reshape(-1,1)).flatten()
        fc_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
        fc_df_lstm = pd.DataFrame({'Date': fc_dates, 'Forecast': lstm_forecast})
        # Plot LSTM forecast
        fig_lstm = go.Figure()
        fig_lstm.add_trace(go.Scatter(x=hist_df.index, y=close_series, name='Historical Close'))
        fig_lstm.add_trace(go.Scatter(x=fc_df_lstm['Date'], y=fc_df_lstm['Forecast'], name='LSTM Forecast', line=dict(dash='dot')))
        fig_lstm.update_layout(title='Historical and LSTM Forecasted Close Prices', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_lstm, use_container_width=True)

