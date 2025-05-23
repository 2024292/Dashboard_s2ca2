import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import altair as alt


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
st.subheader(f'Name: {information["longName"]}')
st.subheader(f'Market Cap: {information["marketCap"]}')
st.subheader(f'Sector: {information["sector"]}')

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

