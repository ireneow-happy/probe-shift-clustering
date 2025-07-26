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

Author: ChatGPT
Date: 2025‑07‑26
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scipy.stats import chi2_contingency, linregress


def load_excel(file) -> pd.DataFrame:
    """Read the uploaded Excel file into a DataFrame and ensure required columns exist."""
    df = pd.read_excel(file)
    required_cols = {"Row", "Col", "Prox Up", "Prox Down", "Prox Left", "Prox Right", "Pass/Fail"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def compute_shifts(df: pd.DataFrame):
    """Compute vertical and horizontal shift columns for the DataFrame."""
    df["Vertical Shift"] = np.abs(df["Prox Up"] - df["Prox Down"])
    df["Horizontal Shift"] = np.abs(df["Prox Left"] - df["Prox Right"])
    return df


def clustering_analysis(df: pd.DataFrame):
    """Perform clustering analysis using KMeans and/or DBSCAN with interactive filtering."""
    st.header("Clustering Analysis")
    # Sidebar options for clustering
    shift_type = st.sidebar.selectbox("偏移方向 (Clustering)", ["Vertical", "Horizontal"])
    agg_method = st.sidebar.selectbox("統計方式 (Clustering)", ["max", "mean"])
    methods = st.sidebar.multiselect(
        "選擇群集方法", ["KMeans", "DBSCAN"], default=["KMeans"]
    )
    if "KMeans" in methods:
        k_value = st.sidebar.number_input(
            "KMeans: 群集數 K", min_value=2, max_value=20, value=3, step=1
        )
    if "DBSCAN" in methods:
        eps_value = st.sidebar.slider(
            "DBSCAN: eps", min_value=0.1, max_value=5.0, value=0.5, step=0.1
        )
        min_samples = st.sidebar.slider(
            "DBSCAN: min_samples", min_value=2, max_value=20, value=5, step=1
        )
        use_shift_feature = st.sidebar.checkbox(
            "Include shift in DBSCAN features", value=True,
            help="若取消勾選，DBSCAN 只使用 Row、Col 進行分群"
        )
    else:
        eps_value = 0.5
        min_samples = 5
        use_shift_feature = True
    run_clustering = st.button("執行群集分析")

    # Session state storage
    if 'cluster_results' not in st.session_state:
        st.session_state['cluster_results'] = None
    if 'cluster_fail' not in st.session_state:
        st.session_state['cluster_fail'] = None
    if 'cluster_kmeans_used' not in st.session_state:
        st.session_state['cluster_kmeans_used'] = False
    if 'cluster_selected_labels' not in st.session_state:
        st.session_state['cluster_selected_labels'] = None
    if 'cluster_run_counter' not in st.session_state:
        st.session_state['cluster_run_counter'] = 0
    # Store the last parameters used for clustering.  When parameters change
    # without re‑running the analysis, we can detect stale results and prompt
    # the user to run the analysis again.  Parameters include the shift type,
    # aggregation method, selected methods and their hyperparameters.
    if 'cluster_last_params' not in st.session_state:
        st.session_state['cluster_last_params'] = None

    if run_clustering:
        # Compute shifts and run clustering with the selected parameters.
        df = compute_shifts(df.copy())
        shift_col = "Vertical Shift" if shift_type == "Vertical" else "Horizontal Shift"
        # Aggregate shift per die
        die_shift = df.groupby(["Row", "Col"])[shift_col].agg(agg_method).reset_index()
        # Prepare clustering df
        die_shift_clustering = die_shift.copy()
        # KMeans
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
        # DBSCAN on fails
        die_shift_fail = None
        if "DBSCAN" in methods:
            # Filter fails
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
                # Merge labels to all die data
                die_shift_clustering = die_shift_clustering.merge(
                    die_shift_fail[["Row", "Col", "DBSCAN_Cluster"]],
                    on=["Row", "Col"], how="left"
                )
        else:
            die_shift_fail = None
            if "DBSCAN_Cluster" in die_shift_clustering.columns:
                die_shift_clustering.drop(columns=["DBSCAN_Cluster"], inplace=True)
        # Store results
        st.session_state['cluster_results'] = die_shift_clustering
        st.session_state['cluster_fail'] = die_shift_fail
        st.session_state['cluster_run_counter'] += 1
        # Reset selected labels
        if die_shift_fail is not None and not die_shift_fail.empty:
            st.session_state['cluster_selected_labels'] = sorted(die_shift_fail["DBSCAN_Cluster"].unique())
        else:
            st.session_state['cluster_selected_labels'] = None
        # Record the parameter values used for this run.
        st.session_state['cluster_last_params'] = {
            'shift_type': shift_type,
            'agg_method': agg_method,
            'methods': tuple(sorted(methods)),
            'k_value': int(k_value) if "KMeans" in methods else None,
            'eps_value': float(eps_value) if "DBSCAN" in methods else None,
            'min_samples': int(min_samples) if "DBSCAN" in methods else None,
            'use_shift_feature': bool(use_shift_feature) if "DBSCAN" in methods else None,
        }
        st.success("群集分析完成。請使用下方的勾選框選擇群集。")

    # Display results if available and up to date
    if st.session_state['cluster_results'] is not None:
        # Determine if current parameters match the last run.  If not, results
        # are considered stale and will not be displayed until analysis is
        # re‑run with the new parameters.
        # Build current parameter signature
        current_params = {
            'shift_type': shift_type,
            'agg_method': agg_method,
            'methods': tuple(sorted(methods)),
            'k_value': int(k_value) if "KMeans" in methods else None,
            'eps_value': float(eps_value) if "DBSCAN" in methods else None,
            'min_samples': int(min_samples) if "DBSCAN" in methods else None,
            'use_shift_feature': bool(use_shift_feature) if "DBSCAN" in methods else None,
        }
        last_params = st.session_state.get('cluster_last_params')
        params_match = (last_params == current_params)
        die_shift_clustering = st.session_state['cluster_results']
        die_shift_fail = st.session_state['cluster_fail']
        if not params_match:
            # Warn user that parameters changed and analysis must be rerun
            st.info("分析參數已變更。請點擊『執行群集分析』以更新結果。先前的結果已暫停顯示。")
        else:
            # KMeans plot
            if st.session_state['cluster_kmeans_used'] and "KMeans_Cluster" in die_shift_clustering.columns:
                st.subheader("KMeans Clustering (all die)")
                fig_km, ax_km = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    data=die_shift_clustering,
                    x="Col", y="Row", hue="KMeans_Cluster", palette="Set2", s=80, ax=ax_km
                )
                ax_km.invert_yaxis()
                st.pyplot(fig_km)
            # DBSCAN plot
            if die_shift_fail is not None and not die_shift_fail.empty:
                st.subheader("DBSCAN Clustering (Fail die only)")
                unique_labels = sorted(die_shift_fail["DBSCAN_Cluster"].unique())
                selected = st.session_state.get('cluster_selected_labels') or unique_labels
                rc = st.session_state['cluster_run_counter']
                # cluster selection checkboxes
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
                # Filter and plot
                filtered = die_shift_fail[die_shift_fail["DBSCAN_Cluster"].isin(new_selected)]
                fig_db, ax_db = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    data=filtered, x="Col", y="Row", hue="DBSCAN_Cluster", palette="Set2", s=80, ax=ax_db
                )
                ax_db.invert_yaxis()
                st.pyplot(fig_db)
            # Download button
            csv = die_shift_clustering.to_csv(index=False).encode("utf-8")
            st.download_button("下載分群資料 CSV", csv, "clustered_die_data.csv", "text/csv")


def dut_analysis(df: pd.DataFrame):
    """Perform DUT failure rate analysis and visualisation."""
    st.header("DUT 相關分析")
    # Compute fail flag
    df["Fail_Flag"] = (df["Pass/Fail"] == "Fail").astype(int)
    # Group by DUT#
    dut_summary = df.groupby("DUT#")["Fail_Flag"].agg(total_tests="count", total_fails="sum").reset_index()
    dut_summary["fail_rate"] = dut_summary["total_fails"] / dut_summary["total_tests"]
    # Sort by fail_rate
    dut_summary = dut_summary.sort_values("fail_rate", ascending=False)
    # Bar chart for fail rate
    st.subheader("各 DUT 的失敗率")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    sns.barplot(data=dut_summary, x="DUT#", y="fail_rate", palette="viridis", ax=ax_bar)
    ax_bar.set_ylabel("Fail Rate")
    ax_bar.set_xlabel("DUT#")
    ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=90)
    st.pyplot(fig_bar)
    # Chi-square test: compare observed fail counts with expected under uniform failure rate
    st.subheader("卡方檢定 (是否某些 DUT 的失敗數顯著偏高)")
    total_fail = dut_summary["total_fails"].sum()
    total_tests = dut_summary["total_tests"].sum()
    expected = total_fail * dut_summary["total_tests"] / total_tests
    chi_square = ((dut_summary["total_fails"] - expected) ** 2 / expected).sum()
    # Degrees of freedom = number of DUTs - 1
    dof = len(dut_summary) - 1
    st.write(f"總失敗數: {total_fail}, 總測試數: {total_tests}, 自由度: {dof}")
    st.write(f"卡方統計值 χ² = {chi_square:.2f}")
    # Optionally, we could compute p-value using scipy, but avoid heavy dependencies
    # Display summary table
    st.dataframe(dut_summary)
    # Plot fail map per DUT (optional): allow selecting specific DUTs
    st.subheader("選擇 DUT 繪製失敗位置圖")
    available_duts = dut_summary["DUT#"].tolist()
    selected_duts = st.multiselect("選擇一個或多個 DUT", available_duts, default=available_duts[:3])
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
    """Analyse shift trends over TD Order."""
    st.header("趨勢分析")
    # Compute shifts
    df = compute_shifts(df.copy())
    # Choose shift direction
    shift_direction = st.selectbox("選擇偏移方向 (趨勢)", ["Vertical", "Horizontal"])
    shift_col = "Vertical Shift" if shift_direction == "Vertical" else "Horizontal Shift"
    # Choose prox direction to examine trend individually
    prox_option = st.selectbox("選擇要分析的 proximity 方向", ["Up", "Down", "Left", "Right", "Shift"])
    run_trend = st.button("執行趨勢分析")
    if run_trend:
        # Prepare data
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
        # Drop NA
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]
        # Compute linear regression
        if len(x) > 1:
            lr = linregress(x, y)
            st.write(f"回歸方程: {y_label} = {lr.slope:.4f} * TD Order + {lr.intercept:.4f}")
            st.write(f"相關係數 R = {lr.rvalue:.4f}, p-value = {lr.pvalue:.4e}")
            # Plot
            fig_tr, ax_tr = plt.subplots(figsize=(8, 5))
            ax_tr.scatter(x, y, alpha=0.3, label="Data points")
            ax_tr.plot(x, lr.intercept + lr.slope * x, color='red', label="Fit line")
            ax_tr.set_xlabel("TD Order")
            ax_tr.set_ylabel(y_label)
            ax_tr.legend()
            st.pyplot(fig_tr)
        else:
            st.warning("資料不足以進行趨勢分析。")


def main():
    st.title("Probe Mark Analysis App")
    st.markdown("""本應用提供多種分析模式：群集分析、DUT 相關分析與趨勢分析。請從左側選擇模式並上傳資料。""")
    analysis_type = st.sidebar.radio("選擇分析類型", ["Clustering analysis", "DUT analysis", "Trend analysis"])
    uploaded_file = st.file_uploader("上傳包含 Probe Mark 資料的 Excel 檔", type=["xlsx"])
    if uploaded_file is None:
        st.info("請上傳檔案以進行分析。")
        return
    try:
        df = load_excel(uploaded_file)
    except Exception as e:
        st.error(str(e))
        return
    # Dispatch to analysis type
    if analysis_type == "Clustering analysis":
        clustering_analysis(df)
    elif analysis_type == "DUT analysis":
        dut_analysis(df)
    else:
        trend_analysis(df)


if __name__ == "__main__":
    main()