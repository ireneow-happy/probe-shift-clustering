import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

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

st.header("Clustering Analysis")
uploaded_file = st.file_uploader("上傳包含 Probe Mark 資料的 Excel 檔", type=["xlsx"], key="clustering_uploaded_file")
if 'clustering_last_uploaded_file' not in st.session_state or st.session_state['clustering_last_uploaded_file'] != uploaded_file:
    st.session_state['cluster_results'] = None
    st.session_state['cluster_fail'] = None
    st.session_state['cluster_kmeans_used'] = False
    st.session_state['cluster_selected_labels'] = None
    st.session_state['cluster_run_counter'] = 0
    st.session_state['clustering_last_uploaded_file'] = uploaded_file
if uploaded_file is None:
    st.info("請上傳檔案以進行分析。")
else:
    try:
        df = load_excel(uploaded_file)
    except Exception as e:
        st.error(str(e))
        st.stop()
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
    option_keys = ["shift_type", "agg_method", "methods", "k_value", "eps_value", "min_samples", "use_shift_feature"]
    for key in option_keys:
        if st.session_state.get(f'last_{key}') != st.session_state.get(key):
            st.session_state['cluster_results'] = None
            st.session_state['cluster_fail'] = None
            st.session_state['cluster_kmeans_used'] = False
            st.session_state['cluster_selected_labels'] = None
            st.session_state['cluster_run_counter'] = 0
            break
    for key in option_keys:
        st.session_state[f'last_{key}'] = st.session_state.get(key)
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
    if run_clustering:
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
    if st.session_state['cluster_results'] is not None:
        die_shift_clustering = st.session_state['cluster_results']
        die_shift_fail = st.session_state['cluster_fail']
        if st.session_state['cluster_kmeans_used'] and "KMeans_Cluster" in die_shift_clustering.columns:
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
            rc = st.session_state['cluster_run_counter']
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