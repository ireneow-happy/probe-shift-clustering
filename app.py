"""
Comprehensive Probe Mark Analysis App

This Streamlit application allows users to perform different types of analysis on
probe mark data stored in an Excel file. Users can choose between:

1. **Clustering analysis**: Uses KMeans and/or DBSCAN to find spatial clusters of
   failing dies, similar to the existing session‑based app. Results are stored in
   session state to allow interactive filtering.
2. **DUT analysis**: Computes failure rates per DUT#, displays a bar chart of
   failure rates, allows mapping fails by DUT location, and performs a chi‑square
   test to highlight DUTs whose failure counts significantly deviate from the
   expected.
3. **Trend analysis**: Examines whether probe mark shift (vertical or horizontal)
   drifts over the TD Order sequence. Produces scatter plots with fitted
   regression lines and shows correlation statistics.

Author: Irene
Date: 2025‑07‑26
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

def load_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    required_cols = {"Row", "Col", "Prox Up", "Prox Down", "Prox Left", "Prox Right", "Pass/Fail", "DUT#", "TD Order"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def compute_shifts(df: pd.DataFrame):
    df["Vertical Shift"] = np.abs(df["Prox Up"] - df["Prox Down"])
    df["Horizontal Shift"] = np.abs(df["Prox Left"] - df["Prox Right"])
    return df

def clustering_analysis(df: pd.DataFrame):
    st.header("Clustering Analysis")
    shift_type = st.sidebar.selectbox("偏移方向 (Clustering)", ["Vertical", "Horizontal"], key="shift_type")
    agg_method = st.sidebar.selectbox("統計方式 (Clustering)", ["max", "mean"], key="agg_method")
    methods = st.sidebar.multiselect(
        "選擇群集方法", ["KMeans", "DBSCAN"], default=["KMeans"], key="methods")
    if "KMeans" in methods:
        k_value = st.sidebar.number_input(
            "KMeans: 群集數 K", min_value=2, max_value=20, value=3, step=1, key="k_value")
    if "DBSCAN" in methods:
        eps_value = st.sidebar.slider(
            "DBSCAN: eps", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="eps_value")
        min_samples = st.sidebar.slider(
            "DBSCAN: min_samples", min_value=2, max_value=20, value=5, step=1, key="min_samples")
        use_shift_feature = st.sidebar.checkbox(
            "Include shift in DBSCAN features", value=True,
            help="若取消勾選，DBSCAN 只使用 Row、Col 進行分群", key="use_shift_feature")
    else:
        eps_value = 0.5
        min_samples = 5
        use_shift_feature = True
    run_clustering = st.button("執行群集分析")
    
    # 簡化檢查邏輯：只在按鈕按下時才檢查選項變動
    if run_clustering:
        # 檢查是否有選項變動，如果有則清空結果
        current_options = (shift_type, agg_method, tuple(methods), k_value, eps_value, min_samples, use_shift_feature)
        if st.session_state.get('last_clustering_options') != current_options:
            st.session_state['cluster_results'] = None
            st.session_state['cluster_fail'] = None
            st.session_state['cluster_kmeans_used'] = False
            st.session_state['cluster_selected_labels'] = None
            st.session_state['cluster_run_counter'] = 0
        st.session_state['last_clustering_options'] = current_options
        
        # 執行分析
        df = compute_shifts(df.copy())
        shift_col = "Vertical Shift" if shift_type == "Vertical" else "Horizontal Shift"
        die_shift = df.groupby(["Row", "Col"])[shift_col].agg(agg_method).reset_index()
        die_shift_clustering = die_shift.copy()
        if "KMeans" in methods:
            st.session_state['cluster_kmeans_used'] = True
            scaler_k = StandardScaler()
            feats = die_shift_clustering[["Col", "Row", shift_col]].values
            feats_scaled = scaler_k.fit_transform(feats)
            km = KMeans(n_clusters=int(k_value), random_state=42)
            km_labels = km.fit_predict(feats_scaled)
            die_shift_clustering["KMeans_Cluster"] = km_labels
        else:
            st.session_state['cluster_kmeans_used'] = False
            if "KMeans_Cluster" in die_shift_clustering.columns:
                die_shift_clustering.drop(columns=["KMeans_Cluster"], inplace=True)
        die_shift_fail = None
        if "DBSCAN" in methods:
            df_fail = df[df["Pass/Fail"] == "Fail"].copy()
            df_fail = compute_shifts(df_fail)
            if df_fail.empty:
                die_shift_fail = pd.DataFrame()
            else:
                die_shift_fail = df_fail.groupby(["Row", "Col"])[shift_col].agg(agg_method).reset_index()
                scaler = StandardScaler()
                if use_shift_feature:
                    feats_fail = die_shift_fail[["Col", "Row", shift_col]].values
                else:
                    feats_fail = die_shift_fail[["Col", "Row"]].values
                feats_scaled_fail = scaler.fit_transform(feats_fail)
                db = DBSCAN(eps=float(eps_value), min_samples=int(min_samples))
                db_labels = db.fit_predict(feats_scaled_fail)
                die_shift_fail["DBSCAN_Cluster"] = db_labels
                die_shift_clustering = die_shift_clustering.merge(
                    die_shift_fail[["Row", "Col", "DBSCAN_Cluster"]],
                    on=["Row", "Col"], how="left"
                )
        else:
            die_shift_fail = None
            if "DBSCAN_Cluster" in die_shift_clustering.columns:
                die_shift_clustering.drop(columns=["DBSCAN_Cluster"], inplace=True)
        st.session_state['cluster_results'] = die_shift_clustering
        st.session_state['cluster_fail'] = die_shift_fail
        st.session_state['cluster_run_counter'] += 1
        if die_shift_fail is not None and not die_shift_fail.empty:
            st.session_state['cluster_selected_labels'] = sorted(die_shift_fail["DBSCAN_Cluster"].unique())
        else:
            st.session_state['cluster_selected_labels'] = None
        st.success("群集分析完成。請使用下方的勾選框選擇群集。")
    
    # 顯示結果（只在有結果時顯示）
    if st.session_state.get('cluster_results') is not None:
        die_shift_clustering = st.session_state['cluster_results']
        die_shift_fail = st.session_state['cluster_fail']
        if st.session_state.get('cluster_kmeans_used') and "KMeans_Cluster" in die_shift_clustering.columns:
            st.subheader("KMeans Clustering (all die)")
            fig_km, ax_km = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=die_shift_clustering,
                x="Col", y="Row", hue="KMeans_Cluster", palette="Set2", s=80, ax=ax_km
            )
            ax_km.invert_yaxis()
            st.pyplot(fig_km)
        if die_shift_fail is not None and not die_shift_fail.empty:
            st.subheader("DBSCAN Clustering (Fail die only)")
            unique_labels = sorted(die_shift_fail["DBSCAN_Cluster"].unique())
            selected = st.session_state.get('cluster_selected_labels') or unique_labels
            rc = st.session_state.get('cluster_run_counter', 0)
            st.markdown("**選擇要顯示的 DBSCAN 群集**")
            new_selected = []
            cols = st.columns(len(unique_labels))
            for idx, lbl in enumerate(unique_labels):
                default_checked = lbl in selected
                key = f"cluster_cb_{lbl}_{rc}"
                with cols[idx]:
                    chk = st.checkbox(str(lbl), value=default_checked, key=key)
                if chk:
                    new_selected.append(lbl)
            st.session_state['cluster_selected_labels'] = new_selected
            filtered = die_shift_fail[die_shift_fail["DBSCAN_Cluster"].isin(new_selected)]
            fig_db, ax_db = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=filtered, x="Col", y="Row", hue="DBSCAN_Cluster", palette="Set2", s=80, ax=ax_db
            )
            ax_db.invert_yaxis()
            st.pyplot(fig_db)
        csv = die_shift_clustering.to_csv(index=False).encode("utf-8")
        st.download_button("下載分群資料 CSV", csv, "clustered_die_data.csv", "text/csv")

def dut_analysis(df: pd.DataFrame):
    st.header("DUT 相關分析")
    run_dut = st.button("執行 DUT 分析")
    if run_dut:
        df["Fail_Flag"] = (df["Pass/Fail"] == "Fail").astype(int)
        dut_summary = df.groupby("DUT#")["Fail_Flag"].agg(total_tests="count", total_fails="sum").reset_index()
        dut_summary["fail_rate"] = dut_summary["total_fails"] / dut_summary["total_tests"]
        dut_summary = dut_summary.sort_values("fail_rate", ascending=False)
        st.session_state['dut_summary'] = dut_summary
        st.session_state['dut_selected_duts'] = dut_summary["DUT#"].tolist()[:3]
        st.success("DUT 分析完成。請於下方選擇DUT繪圖。")
    if st.session_state.get('dut_summary') is not None:
        dut_summary = st.session_state['dut_summary']
        st.subheader("各 DUT 的失敗率")
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        sns.barplot(data=dut_summary, x="DUT#", y="fail_rate", palette="viridis", ax=ax_bar)
        ax_bar.set_ylabel("Fail Rate")
        ax_bar.set_xlabel("DUT#")
        ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=90)
        st.pyplot(fig_bar)
        st.subheader("卡方檢定 (是否某些 DUT 的失敗數顯著偏高)")
        total_fail = dut_summary["total_fails"].sum()
        total_tests = dut_summary["total_tests"].sum()
        expected = total_fail * dut_summary["total_tests"] / total_tests
        chi_square = ((dut_summary["total_fails"] - expected) ** 2 / expected).sum()
        dof = len(dut_summary) - 1
        st.write(f"總失敗數: {total_fail}, 總測試數: {total_tests}, 自由度: {dof}")
        st.write(f"卡方統計值 χ² = {chi_square:.2f}")
        st.dataframe(dut_summary)
        st.subheader("選擇 DUT 繪製失敗位置圖")
        available_duts = dut_summary["DUT#"].tolist()
        
        # 簡化邏輯：直接使用 multiselect，讓 Streamlit 自動管理狀態
        selected_duts = st.multiselect("選擇一個或多個 DUT", available_duts, key="dut_selected_duts")
        
        # 如果沒有選擇，使用前三個 DUT
        if not selected_duts and available_duts:
            selected_duts = available_duts[:3]
        
        if selected_duts:
            df_fail = df[df["Pass/Fail"] == "Fail"]
            fig_map, ax_map = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df_fail[df_fail["DUT#"].isin(selected_duts)],
                x="Col", y="Row", hue="DUT#", palette="tab10", s=60, ax=ax_map
            )
            ax_map.invert_yaxis()
            st.pyplot(fig_map)

def trend_analysis(df: pd.DataFrame):
    st.header("趨勢分析")
    shift_direction = st.selectbox("選擇偏移方向 (趨勢)", ["Vertical", "Horizontal"], key="trend_shift_direction")
    prox_option = st.selectbox("選擇要分析的 proximity 方向", ["Up", "Down", "Left", "Right", "Shift"], key="trend_prox_option")
    run_trend = st.button("執行趨勢分析")
    
    if run_trend:
        # 檢查選項是否變動
        current_trend_options = (shift_direction, prox_option)
        if st.session_state.get('last_trend_options') != current_trend_options:
            st.session_state['trend_result'] = None
        st.session_state['last_trend_options'] = current_trend_options
        
        df = compute_shifts(df.copy())
        shift_col = "Vertical Shift" if shift_direction == "Vertical" else "Horizontal Shift"
        df_fail = df[df["Pass/Fail"] == "Fail"].copy()
        if prox_option == "Shift":
            y = df_fail[shift_col]
            y_label = shift_col
        else:
            if prox_option == "Up":
                y = df_fail["Prox Up"]
            elif prox_option == "Down":
                y = df_fail["Prox Down"]
            elif prox_option == "Left":
                y = df_fail["Prox Left"]
            else:
                y = df_fail["Prox Right"]
            y_label = f"Prox {prox_option}"
        x = df_fail["TD Order"]
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        if len(x) > 1:
            lr = linregress(x, y)
            st.session_state['trend_result'] = {
                'x': x,
                'y': y,
                'slope': lr.slope,
                'intercept': lr.intercept,
                'rvalue': lr.rvalue,
                'pvalue': lr.pvalue,
                'y_label': y_label
            }
            st.success("趨勢分析完成。")
        else:
            st.session_state['trend_result'] = None
            st.warning("資料不足以進行趨勢分析。")
    if st.session_state.get('trend_result') is not None:
        res = st.session_state['trend_result']
        st.write(f"回歸方程: {res['y_label']} = {res['slope']:.4f} * TD Order + {res['intercept']:.4f}")
        st.write(f"相關係數 R = {res['rvalue']:.4f}, p-value = {res['pvalue']:.4e}")
        fig_tr, ax_tr = plt.subplots(figsize=(8, 5))
        ax_tr.scatter(res['x'], res['y'], alpha=0.3, label="Data points")
        ax_tr.plot(res['x'], res['intercept'] + res['slope'] * res['x'], color='red', label="Fit line")
        ax_tr.set_xlabel("TD Order")
        ax_tr.set_ylabel(res['y_label'])
        ax_tr.legend()
        st.pyplot(fig_tr)

def main():
    st.title("Probe Mark Analysis App")
    st.markdown("本應用提供多種分析模式：群集分析、DUT 相關分析與趨勢分析。請從左側選擇模式並上傳資料。")
    analysis_type = st.sidebar.radio("選擇分析類型", ["Clustering analysis", "DUT analysis", "Trend analysis"])
    uploaded_file = st.file_uploader("上傳包含 Probe Mark 資料的 Excel 檔", type=["xlsx"], key="uploaded_file")
    
    # 簡化檔案變動檢查
    if uploaded_file is not None and st.session_state.get('last_uploaded_file') != uploaded_file:
        # 清空所有分析結果
        for key in ['cluster_results', 'cluster_fail', 'cluster_kmeans_used', 'cluster_selected_labels', 
                   'cluster_run_counter', 'dut_summary', 'dut_selected_duts', 'trend_result']:
            st.session_state[key] = None
        st.session_state['last_uploaded_file'] = uploaded_file
    
    if uploaded_file is None:
        st.info("請上傳檔案以進行分析。")
        return
    try:
        df = load_excel(uploaded_file)
    except Exception as e:
        st.error(str(e))
        return
    if analysis_type == "Clustering analysis":
        clustering_analysis(df)
    elif analysis_type == "DUT analysis":
        dut_analysis(df)
    else:
        trend_analysis(df)

if __name__ == "__main__":
    main()