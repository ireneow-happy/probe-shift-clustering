import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Probe Mark Analysis App")

# Sidebar
st.sidebar.header("選擇分析類型")
analysis_type = st.sidebar.radio(
    "分析類型",
    ["Clustering analysis", "DUT analysis", "Trend analysis"],
    captions=[
        "使用 KMeans 和 DBSCAN 對 Fail die 做群集分析",
        "分析 Fail 對 DUT 的關係，例如 Fail rate、空間分佈與統計檢定",
        "分析 Fail 順序與偏移變化趨勢，例如是否隨 TD 順序偏移"
    ]
)

uploaded_file = st.file_uploader("上傳包含 Probe Mark 資料的 Excel 檔", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state["data"] = df

# CLUSTERING SECTION
if analysis_type == "Clustering analysis" and "data" in st.session_state:
    df = st.session_state["data"]
    st.header("Clustering Analysis")
    st.markdown("請設定以下參數，點擊下方按鈕以執行群集分析。")

    with st.sidebar:
        direction = st.selectbox("偏移方向 (Clustering)", ["Vertical", "Horizontal"], disabled=False)
        method = st.selectbox("統計方式 (Clustering)", ["max", "mean"], disabled=False)
        selected_methods = st.multiselect("選擇群集方法", ["KMeans", "DBSCAN"], default=["KMeans", "DBSCAN"])
        k = st.number_input("KMeans: 群集數 K", min_value=1, max_value=10, value=3)
        eps = st.slider("DBSCAN: eps", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.slider("DBSCAN: min_samples", 2, 20, 5)
        use_shift = st.checkbox("Include shift in DBSCAN features")

    if st.button("執行群集分析"):
        if "Pass/Fail" not in df.columns:
            st.error("資料中找不到 'Pass/Fail' 欄位，請確認 Excel 格式正確。")
        else:
            df_fail = df[df["Pass/Fail"] == "Fail"]
            if df_fail.empty:
                st.warning("資料中找不到任何 Fail 樣本，請檢查 'Pass/Fail' 欄位內容。")
            else:
                features = ["Col", "Row"]
                if use_shift:
                    if direction == "Vertical":
                        features.append("Prox Down")
                    elif direction == "Horizontal":
                        features.append("Prox Right")

                X = df_fail[features].fillna(0)

                if "KMeans" in selected_methods:
                    kmeans = KMeans(n_clusters=k, n_init="auto")
                    df_fail["KMeans_Cluster"] = kmeans.fit_predict(X)

                if "DBSCAN" in selected_methods:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    df_fail["DBSCAN_Cluster"] = dbscan.fit_predict(X)

                st.session_state["df_fail"] = df_fail

    if "df_fail" in st.session_state:
        df_fail = st.session_state["df_fail"]
        st.subheader("DBSCAN Clustering (Fail die only)")

        clusters = sorted(df_fail["DBSCAN_Cluster"].unique())
        selected_clusters = st.multiselect("顯示哪些 DBSCAN 群集", clusters, default=clusters)
        filtered = df_fail[df_fail["DBSCAN_Cluster"].isin(selected_clusters)]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered, x="Col", y="Row", hue="DBSCAN_Cluster", ax=ax, palette="tab10")
        ax.invert_yaxis()
        st.pyplot(fig)
