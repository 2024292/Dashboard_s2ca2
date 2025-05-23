import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

# 股票公司列表
companies = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
models_dir = "models"  # 模型存放路径
data_dir = "data"      # 数据文件夹

# 日期预测选项
predict_days = st.sidebar.selectbox("📅 Select prediction horizon", ["1 day", "3 days", "7 days"])
day_num = int(predict_days.split()[0])

# 选择公司
selected_company = st.sidebar.selectbox("🏢 Select a company", companies)

# 加载历史数据
@st.cache_data
def load_data(company):
    path = os.path.join(data_dir, f"{company}_data.csv")
    return pd.read_csv(path, parse_dates=["Date"])

# 加载模型并进行预测
def load_model_and_predict(company, horizon):
    model_path = os.path.join(models_dir, f"{company}_model_{horizon}.pkl")
    model = joblib.load(model_path)
    
    # 加载特征（这里假设特征预处理也已完成）
    feature_path = os.path.join(data_dir, f"{company}_features_{horizon}.csv")
    features = pd.read_csv(feature_path)
    
    prediction = model.predict(features)
    return prediction

# 显示历史收盘价
df = load_data(selected_company)
st.subheader(f"📊 {selected_company} Historical Close Price")
fig = px.line(df, x="Date", y="Close", title=f"{selected_company} Close Price Over Time")
st.plotly_chart(fig, use_container_width=True)

# 加载并显示预测
try:
    prediction = load_model_and_predict(selected_company, day_num)
    st.success(f"✅ Predicted Close Price for next {day_num} day(s): {prediction[-1]:.2f}")
    
    # 显示预测趋势（最后一段历史数据 + 预测）
    combined = df[["Date", "Close"]].copy()
    last_date = combined["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=day_num)
    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Close": prediction
    })
    combined = pd.concat([combined, pred_df], ignore_index=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=combined["Date"], y=combined["Close"], mode="lines+markers", name="Price"))
    fig2.update_layout(title=f"{selected_company} Close Price Forecast ({predict_days})", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"❌ Could not load model or prediction data for {selected_company} ({day_num} days). Please check files.")
    st.exception(e)

# 展示EDA图像（如果有）
eda_path = os.path.join(data_dir, f"{selected_company}_eda.png")
if os.path.exists(eda_path):
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    st.image(eda_path, use_column_width=True)