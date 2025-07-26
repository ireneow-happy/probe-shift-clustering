"""
Streamlit app for probe mark shift analysis with interactive DBSCAN cluster filtering.

This version improves on earlier implementations by storing intermediate results
in Streamlit's session state so that users can interactively filter DBSCAN
clusters via checkboxes without having to re‑run the entire analysis.  The
analysis (reading the Excel file, computing shifts, aggregating per die and
running KMeans/DBSCAN) is performed once when the user clicks the
"🚀 執行分析" button.  Results are cached in ``st.session_state`` and reused
for plotting and filtering.  When the user adjusts the cluster checkboxes or
other widgets, the app reruns but uses the cached results, so the chart
updates immediately.  Selecting a different ``eps`` or ``min_samples`` and
re‑running the analysis recomputes DBSCAN and updates the stored results.

Key features:

* Computes vertical or horizontal probe‑mark shift from proximal distances.
* Aggregates shift by die (Row, Col) using max or mean.
* Generates a heatmap of all dies and KMeans clustering (optional).
* Filters fails and performs DBSCAN clustering only on failing dies.
* Stores computed results in session state for interactive use.
* Allows users to toggle visibility of individual DBSCAN clusters via
  checkboxes; selections persist across reruns.

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


def main():
    st.title("Probe Mark Shift Clustering App (Interactive Session)")

    # Description
    st.markdown(
        """
        **📘 計算邏輯與功能說明**

        - 偏移量（偏心度）計算公式：
          - 垂直偏移 = |Prox Up − Prox Down|
          - 水平偏移 = |Prox Left − Prox Right|

        - 本工具可支援：
          - 垂直或水平偏移分析（可選）
          - 使用最大值或平均值分析（可選）
          - 對選定偏移類型產生熱力圖與分群結果
          - 支援 KMeans 與 DBSCAN 分群
          - 匯出含分群資訊的結果檔

        **互動功能**：分析結果會被儲存在 session state 中。當您
        勾選/取消 DBSCAN 群集、調整顯示設定時，圖表會立即更新，
        不需要重新跑整個分析。若調整了 DBSCAN 參數 (eps、min_samples)，
        請再次按下「🚀 執行分析」重新計算。
        """
    )

    # Sidebar for parameters
    st.sidebar.header("分析參數")
    shift_type = st.sidebar.selectbox("選擇偏移方向", ["Vertical", "Horizontal"])
    agg_method = st.sidebar.selectbox("選擇統計方式", ["max", "mean"])

    st.sidebar.header("Clustering Settings")
    model_selection = st.sidebar.multiselect(
        "Select Clustering Methods", ["KMeans", "DBSCAN"], default=["KMeans"]
    )
    if "KMeans" in model_selection:
        k_value = st.sidebar.number_input(
            "KMeans: Number of Clusters (K)", min_value=2, max_value=20, value=3, step=1
        )
    if "DBSCAN" in model_selection:
        eps_value = st.sidebar.slider(
            "DBSCAN: eps (neighborhood size)", min_value=0.1, max_value=5.0, value=0.5, step=0.1
        )
        min_samples = st.sidebar.slider(
            "DBSCAN: min_samples", min_value=2, max_value=20, value=5, step=1
        )
    else:
        eps_value = 0.5
        min_samples = 5

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    run_button = st.button("🚀 執行分析")

    # Initialize session state variables
    if 'analysis_ready' not in st.session_state:
        st.session_state['analysis_ready'] = False
    if 'die_shift_clustering' not in st.session_state:
        st.session_state['die_shift_clustering'] = None
    if 'die_shift_fail' not in st.session_state:
        st.session_state['die_shift_fail'] = None
    if 'heatmap_data' not in st.session_state:
        st.session_state['heatmap_data'] = None
    if 'shift_label' not in st.session_state:
        st.session_state['shift_label'] = None
    if 'agg_method_used' not in st.session_state:
        st.session_state['agg_method_used'] = None
    if 'kmeans_used' not in st.session_state:
        st.session_state['kmeans_used'] = False
    if 'selected_clusters' not in st.session_state:
        st.session_state['selected_clusters'] = None
    if 'run_counter' not in st.session_state:
        st.session_state['run_counter'] = 0

    # Perform analysis when run button is pressed
    if uploaded_file is not None and run_button:
        try:
            df = pd.read_excel(uploaded_file)
            required_cols = {"Prox Up", "Prox Down", "Prox Left", "Prox Right", "Row", "Col"}
            if not required_cols.issubset(df.columns):
                st.error(f"Missing columns: {required_cols.difference(df.columns)}")
            else:
                # Compute shift columns
                df["Vertical Shift"] = np.abs(df["Prox Up"] - df["Prox Down"])
                df["Horizontal Shift"] = np.abs(df["Prox Left"] - df["Prox Right"])
                shift_column = "Vertical Shift" if shift_type == "Vertical" else "Horizontal Shift"
                shift_label = "垂直" if shift_type == "Vertical" else "水平"
                st.session_state['shift_label'] = shift_label
                st.session_state['agg_method_used'] = agg_method

                # Aggregate shift per die
                die_shift = df.groupby(["Row", "Col"])[shift_column].agg(agg_method).reset_index()

                # Prepare clustering DataFrame
                die_shift_clustering = die_shift.copy()

                # KMeans clustering if selected
                if "KMeans" in model_selection:
                    st.session_state['kmeans_used'] = True
                    scaler_k = StandardScaler()
                    features_k = die_shift_clustering[["Col", "Row", shift_column]].values
                    features_scaled_k = scaler_k.fit_transform(features_k)
                    kmeans = KMeans(n_clusters=int(k_value), random_state=42)
                    clusters_kmeans = kmeans.fit_predict(features_scaled_k)
                    die_shift_clustering["KMeans_Cluster"] = clusters_kmeans
                else:
                    st.session_state['kmeans_used'] = False
                    if "KMeans_Cluster" in die_shift_clustering.columns:
                        die_shift_clustering.drop(columns=["KMeans_Cluster"], inplace=True)

                # DBSCAN clustering on fails if selected
                die_shift_fail = None
                if "DBSCAN" in model_selection:
                    if "Pass/Fail" not in df.columns:
                        st.error("Pass/Fail column not found in the uploaded file. Cannot perform DBSCAN on fails.")
                    else:
                        df_fail = df[df["Pass/Fail"] == "Fail"].copy()
                        if df_fail.empty:
                            st.warning("No 'Fail' records found. DBSCAN clustering skipped.")
                            die_shift_fail = pd.DataFrame()
                        else:
                            df_fail["Vertical Shift"] = np.abs(df_fail["Prox Up"] - df_fail["Prox Down"])
                            df_fail["Horizontal Shift"] = np.abs(df_fail["Prox Left"] - df_fail["Prox Right"])
                            die_shift_fail = df_fail.groupby(["Row", "Col"])[shift_column].agg(agg_method).reset_index()
                            scaler_db = StandardScaler()
                            features_fail = die_shift_fail[["Col", "Row", shift_column]].values
                            features_scaled_fail = scaler_db.fit_transform(features_fail)
                            dbscan = DBSCAN(eps=float(eps_value), min_samples=int(min_samples))
                            clusters_dbscan = dbscan.fit_predict(features_scaled_fail)
                            die_shift_fail["DBSCAN_Cluster"] = clusters_dbscan
                            # Merge clusters back to full die table
                            die_shift_clustering = die_shift_clustering.merge(
                                die_shift_fail[["Row", "Col", "DBSCAN_Cluster"]],
                                on=["Row", "Col"],
                                how="left",
                            )
                else:
                    die_shift_fail = None
                    if "DBSCAN_Cluster" in die_shift_clustering.columns:
                        die_shift_clustering.drop(columns=["DBSCAN_Cluster"], inplace=True)

                # Store heatmap data
                heatmap_data = die_shift.pivot(index="Row", columns="Col", values=shift_column)
                heatmap_data = heatmap_data.sort_index(ascending=True).sort_index(axis=1, ascending=True)
                st.session_state['heatmap_data'] = heatmap_data

                # Save results to session state
                st.session_state['die_shift_clustering'] = die_shift_clustering
                st.session_state['die_shift_fail'] = die_shift_fail
                st.session_state['analysis_ready'] = True
                # Increment run counter for checkbox keys
                st.session_state['run_counter'] += 1
                # Initialize selected clusters based on previous state or default to all clusters
                if die_shift_fail is not None and not die_shift_fail.empty:
                    unique_labels = sorted(die_shift_fail["DBSCAN_Cluster"].unique())
                    prev_selected = st.session_state.get('selected_clusters')
                    if prev_selected is None:
                        st.session_state['selected_clusters'] = unique_labels
                    else:
                        # Keep intersection of previous selection and new labels
                        st.session_state['selected_clusters'] = [l for l in unique_labels if l in prev_selected]
                else:
                    st.session_state['selected_clusters'] = None

                st.success("分析完成，結果已更新。您可以在下方互動式選擇群集並查看圖表。")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Display results if analysis has been run at least once
    if st.session_state['analysis_ready']:
        heatmap_data = st.session_state['heatmap_data']
        shift_label = st.session_state['shift_label'] or ''
        agg_used = st.session_state['agg_method_used'] or agg_method
        # Show heatmap
        if heatmap_data is not None:
            st.subheader(f"{shift_label}偏移量熱力圖 ({'最大值' if agg_used == 'max' else '平均值'})")
            fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax_hm)
            ax_hm.set_title(f"{shift_label}偏移熱力圖 ({agg_used})")
            st.pyplot(fig_hm)

        # Show KMeans clustering if it was computed
        if st.session_state['kmeans_used'] and st.session_state['die_shift_clustering'] is not None and "KMeans_Cluster" in st.session_state['die_shift_clustering'].columns:
            st.subheader("KMeans Clustering (all die)")
            fig_km, ax_km = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=st.session_state['die_shift_clustering'],
                x="Col",
                y="Row",
                hue="KMeans_Cluster",
                palette="Set2",
                s=100,
                ax=ax_km,
            )
            ax_km.invert_yaxis()
            st.pyplot(fig_km)

        # Show DBSCAN results with interactive filtering
        die_shift_fail = st.session_state['die_shift_fail']
        if die_shift_fail is not None and not die_shift_fail.empty:
            # K-distance plot to help choose eps
            st.subheader("📐 K-distance Plot for DBSCAN (Fail data)")
            # Use the same shift column (there is only one shift column in die_shift_fail besides Row/Col/Cluster)
            shift_cols = [c for c in die_shift_fail.columns if c.endswith("Shift")]  # e.g. Vertical Shift or Horizontal Shift
            features = die_shift_fail[["Col", "Row", shift_cols[0]]].values if shift_cols else die_shift_fail[["Col", "Row"]].values
            scaler_kdist = StandardScaler()
            features_scaled = scaler_kdist.fit_transform(features)
            nbrs = NearestNeighbors(n_neighbors=max(1, int(min_samples))).fit(features_scaled)
            distances, _ = nbrs.kneighbors(features_scaled)
            k_distances = np.sort(distances[:, -1])
            knee_locator = KneeLocator(
                range(len(k_distances)), k_distances, curve="convex", direction="increasing"
            )
            elbow_eps = k_distances[knee_locator.knee] if knee_locator.knee else None
            fig_kd, ax_kd = plt.subplots(figsize=(10, 4))
            ax_kd.plot(k_distances, label="K-distance")
            if elbow_eps:
                ax_kd.axhline(y=elbow_eps, color="red", linestyle="--", label=f"Suggested eps ≈ {elbow_eps:.2f}")
            ax_kd.set_title(f"K-distance plot (min_samples={min_samples})")
            ax_kd.set_ylabel(f"Distance to {min_samples}-th nearest neighbor")
            ax_kd.set_xlabel("Points sorted by distance")
            ax_kd.legend()
            st.pyplot(fig_kd)

            # Interactive cluster selection
            unique_labels = sorted(die_shift_fail["DBSCAN_Cluster"].unique())
            prev_selected = st.session_state.get('selected_clusters') or unique_labels
            selected_clusters = []
            st.markdown("**選擇要顯示的 DBSCAN 群集** (透過勾選方塊，可複選)\n")
            cols = st.columns(len(unique_labels)) if unique_labels else [st]
            # Use run_counter to create unique keys per run so that value parameter is respected
            rc = st.session_state['run_counter']
            for idx, label in enumerate(unique_labels):
                default_checked = label in prev_selected
                key = f"cluster_cb_{label}_{rc}"
                with cols[idx]:
                    checked = st.checkbox(str(label), value=default_checked, key=key)
                if checked:
                    selected_clusters.append(label)
            # Persist selection
            st.session_state['selected_clusters'] = selected_clusters
            # Filter and plot
            filtered = die_shift_fail[die_shift_fail["DBSCAN_Cluster"].isin(selected_clusters)]
            st.subheader("DBSCAN Clustering (Fail die only)")
            fig_db, ax_db = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=filtered,
                x="Col",
                y="Row",
                hue="DBSCAN_Cluster",
                palette="Set2",
                s=100,
                ax=ax_db,
            )
            ax_db.invert_yaxis()
            st.pyplot(fig_db)

        # Download button for full aggregated data
        die_shift_clustering = st.session_state['die_shift_clustering']
        if die_shift_clustering is not None:
            csv = die_shift_clustering.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Clustered Data as CSV",
                csv,
                "clustered_die_data.csv",
                "text/csv",
            )

    elif uploaded_file is None:
        st.info("Please upload a probe mark Excel file and click '執行分析' to begin.")


if __name__ == "__main__":
    main()