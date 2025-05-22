import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model

# 设置页面
st.set_page_config(page_title="📈 Stock Forecast Dashboard", layout="wide")
st.title("📊 LSTM Stock Price Forecast Dashboard")

# ----------- 工具函数 ------------
def download_file_from_drive(file_id, output_path):
    """从 Google Drive 下载文件（如果本地没有）"""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        
        gdown.download(url, output_path, quiet=False)
    return output_path

def load_lstm_model(model_path):
    """加载 .keras 模型"""
    return load_model(model_path)

def load_test_data(csv_path):
    """加载测试数据 CSV"""
    return pd.read_csv(csv_path)

# ----------- 模型 & 数据映射 ------------

MODEL_MAP = {
    "AAPL - T+1": {
        "model_id": "10mztuP5q8uc47bjvm4DE1E1lKDHksPyx",  # 替换为Google Drive的ID
        "data_id": "1jRaeRGuN4FLQWXcGNIVRpD7-VSrLWMpn"
    },
    "AAPL - T+3": {
        "model_id": "1D-VQEuB05mrf-UuYeQEEV1xHvffK6b0K",
        "data_id": "1jRaeRGuN4FLQWXcGNIVRpD7-VSrLWMpn"
    },
    # 可继续添加更多模型
}

# ----------- 用户选择模型 ------------

selected_model = st.selectbox("请选择模型进行预测", list(MODEL_MAP.keys()))

if selected_model:
    model_info = MODEL_MAP[selected_model]
    model_file = f"{selected_model.replace(' ', '_')}_model.keras"
    data_file = f"{selected_model.replace(' ', '_')}_test.csv"

    # 下载 & 加载模型
    with st.spinner("加载模型中..."):
        model_path = download_file_from_drive(model_info["model_id"], model_file)
        model = load_lstm_model(model_path)

    # 下载 & 加载数据
    with st.spinner("加载测试数据..."):
        data_path = download_file_from_drive(model_info["data_id"], data_file)
        df = load_test_data(data_path)

    st.success("模型和数据加载成功！")

    # ----------- 构建 X 并预测 ------------
    # 注意：假设你已经上传了一个 'X.npy' 预处理好的特征文件
    x_file = f"{selected_model.replace(' ', '_')}_X.npy"
    if os.path.exists(x_file):
        X = np.load(x_file)
        preds = model.predict(X).flatten()
        actual = df["Close"].values[-len(preds):]

        # ----------- 可视化 ------------
        st.subheader("📉 实际 vs 预测")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(actual, label="实际价格")
        ax.plot(preds, label="预测价格")
        ax.set_title(f"{selected_model} - 预测 vs 实际")
        ax.set_xlabel("时间步")
        ax.set_ylabel("收盘价")
        ax.legend()
        st.pyplot(fig)

        st.metric("📍 最后预测值", f"{preds[-1]:.2f}")
    else:
        st.warning("⚠️ 找不到预处理特征文件（X.npy），请先构建并上传到同一目录。")