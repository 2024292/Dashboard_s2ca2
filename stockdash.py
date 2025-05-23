import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import altair as alt

def fetch_stock_info(symbol):
    pass


def fetch_quarterly_financials(symbol):
    stock = yf.Ticker(symbol)
    quarterly_financials = stock.quarterly_financials
    return quarterly_financials


def fetch_annual_financials(symbol):
    stock = yf.Ticker(symbol)
    annual_financials = stock.financials
    return annual_financials

def fetch_weekly_price_history(symbol):
    stock = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=52)  # Fetch data for the last year
    weekly_data = stock.history(period='1y', interval='1wk')
    return weekly_data

st.title("Stock Dashboard")
symbol = st.text_input("Enter stock symbol (e.g., AAPL, TSLA):", "AAPL")