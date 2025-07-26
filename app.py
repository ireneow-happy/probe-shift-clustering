"""
This Streamlit app is a modified version of a probe‑mark shift
analysis tool.  It reads a wafer‑level Excel spreadsheet, computes
vertical or horizontal probe‑mark shifts from proximal distances and
generates heatmaps for visual inspection.  For clustering, both
KMeans and DBSCAN are offered, but the DBSCAN section has been
rewritten to focus on *only failing die* and to avoid the typical
pitfall of merging the entire wafer into a single cluster.  The
original implementation clustered every die on the wafer (pass and
fail) using an `eps` value of 1 or larger, which caused DBSCAN to
report just one cluster.  This version instead filters the input
DataFrame for entries where the ``Pass/Fail`` column equals
``"Fail"``, recalculates the shift metric on that subset, groups by
``Row`` and ``Col``, and uses those positions (with the shift
magnitude as an additional feature) to build a DBSCAN model.  The
default ``eps`` slider now starts at 0.5, which is appropriate for
integer row/column coordinates where adjacent die are exactly one unit
apart.  Users can interactively adjust ``eps`` and ``min_samples`` to
explore cluster formation.  The resulting cluster labels are merged
back into the full aggregated die table for easy export.

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

# Set Streamlit page configuration
st.title("Probe Mark Shift Clustering App (DBSCAN Fixed)")

# Markdown to explain the logic
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

    **重要調整**：為了讓 DBSCAN 能辨識出失敗品的群聚區域，程式會先依
    ``Pass/Fail`` 欄位篩選出 ``Fail`` 的紀錄再進行分群，避免全圖
    被視為一個單一群集。原始資料的熱力圖仍包含所有 die 以便比對。
    """
)

# Sidebar for user inputs
st.sidebar.header("分析參數")
shift_type = st.sidebar.selectbox("選擇偏移方向", ["Vertical", "Horizontal"])
agg_method = st.sidebar.selectbox("選擇統計方式", ["max", "mean"])

st.sidebar.header("Clustering Settings")
model_selection = st.sidebar.multiselect(
    "Select Clustering Methods",
    ["KMeans", "DBSCAN"],
    default=["KMeans"]
)

if "KMeans" in model_selection:
    k_value = st.sidebar.number_input(
        "KMeans: Number of Clusters (K)", min_value=2, max_value=20, value=3, step=1
    )
if "DBSCAN" in model_selection:
    # default eps lowered to 0.5; min value at 0.1 with step 0.1
    eps_value = st.sidebar.slider(
        "DBSCAN: eps (neighborhood size)", min_value=0.1, max_value=5.0, value=0.5, step=0.1
    )
    min_samples = st.sidebar.slider(
        "DBSCAN: min_samples", min_value=2, max_value=20, value=5, step=1
    )

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
run_analysis = st.button("🚀 執行分析")

if uploaded_file is not None and run_analysis:
    try:
        df = pd.read_excel(uploaded_file)
        # Validate that the necessary columns exist
        required_cols = {"Prox Up", "Prox Down", "Prox Left", "Prox Right", "Row", "Col"}
        if not required_cols.issubset(df.columns):
            st.error(f"Missing columns: {required_cols.difference(df.columns)}")
        else:
            # Compute shift columns
            df["Vertical Shift"] = np.abs(df["Prox Up"] - df["Prox Down"])
            df["Horizontal Shift"] = np.abs(df["Prox Left"] - df["Prox Right"])

            # Determine shift type and label
            shift_column = "Vertical Shift" if shift_type == "Vertical" else "Horizontal Shift"
            shift_label = "垂直" if shift_type == "Vertical" else "水平"

            # Aggregate shift per die (Row, Col) using selected aggregation function
            die_shift = df.groupby(["Row", "Col"])[shift_column].agg(agg_method).reset_index()

            # Heatmap of all die positions
            st.subheader(f"{shift_label}偏移量熱力圖 ({'最大值' if agg_method == 'max' else '平均值'})")
            heatmap_data = die_shift.pivot(index="Row", columns="Col", values=shift_column)
            heatmap_data = heatmap_data.sort_index(ascending=True).sort_index(axis=1, ascending=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
            ax.set_title(f"{shift_label}偏移熱力圖 ({agg_method})")
            st.pyplot(fig)

            # Prepare clustering DataFrame (copy) for merging cluster labels later
            die_shift_clustering = die_shift.copy()
            # For KMeans, we include all die (pass and fail)
            if "KMeans" in model_selection:
                # Standardize features: use Col, Row and shift_column
                scaler_k = StandardScaler()
                features_k = die_shift_clustering[["Col", "Row", shift_column]].values
                features_scaled_k = scaler_k.fit_transform(features_k)
                kmeans = KMeans(n_clusters=int(k_value), random_state=42)
                clusters_kmeans = kmeans.fit_predict(features_scaled_k)
                die_shift_clustering["KMeans_Cluster"] = clusters_kmeans
                # Plot KMeans result
                st.subheader("KMeans Clustering (all die)")
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    data=die_shift_clustering,
                    x="Col",
                    y="Row",
                    hue="KMeans_Cluster",
                    palette="Set2",
                    s=100,
                    ax=ax1,
                )
                ax1.invert_yaxis()
                st.pyplot(fig1)

            # DBSCAN clustering on fail-only data
            if "DBSCAN" in model_selection:
                if "Pass/Fail" not in df.columns:
                    st.error("Pass/Fail column not found in the uploaded file. Cannot perform DBSCAN on fails.")
                else:
                    # Filter only fail records
                    df_fail = df[df["Pass/Fail"] == "Fail"].copy()
                    if df_fail.empty:
                        st.warning("No 'Fail' records found. DBSCAN clustering skipped.")
                    else:
                        # Compute shift columns for fail subset
                        df_fail["Vertical Shift"] = np.abs(df_fail["Prox Up"] - df_fail["Prox Down"])
                        df_fail["Horizontal Shift"] = np.abs(df_fail["Prox Left"] - df_fail["Prox Right"])
                        # Group by die
                        die_shift_fail = df_fail.groupby(["Row", "Col"])[shift_column].agg(agg_method).reset_index()
                        # Scale features: use Col, Row and shift value
                        scaler_db = StandardScaler()
                        features_fail = die_shift_fail[["Col", "Row", shift_column]].values
                        features_scaled_fail = scaler_db.fit_transform(features_fail)

                        # K-distance plot for fail data
                        st.subheader("📐 K-distance Plot for DBSCAN (Fail data)")
                        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(features_scaled_fail)
                        distances, _ = nbrs.kneighbors(features_scaled_fail)
                        k_distances = np.sort(distances[:, -1])
                        knee_locator = KneeLocator(
                            range(len(k_distances)), k_distances, curve="convex", direction="increasing"
                        )
                        elbow_eps = k_distances[knee_locator.knee] if knee_locator.knee else None
                        fig_k, ax_k = plt.subplots(figsize=(10, 4))
                        ax_k.plot(k_distances, label="K-distance")
                        if elbow_eps:
                            ax_k.axhline(y=elbow_eps, color="red", linestyle="--", label=f"Suggested eps ≈ {elbow_eps:.2f}")
                        ax_k.set_title(f"K-distance plot (min_samples={min_samples})")
                        ax_k.set_ylabel(f"Distance to {min_samples}-th nearest neighbor")
                        ax_k.set_xlabel("Points sorted by distance")
                        ax_k.legend()
                        st.pyplot(fig_k)

                        # Fit DBSCAN on fail data
                        dbscan = DBSCAN(eps=float(eps_value), min_samples=int(min_samples))
                        clusters_dbscan = dbscan.fit_predict(features_scaled_fail)
                        die_shift_fail["DBSCAN_Cluster"] = clusters_dbscan

                        # Allow users to filter specific clusters for display
                        # Collect unique cluster labels (including noise labeled as -1)
                        unique_labels = sorted(die_shift_fail["DBSCAN_Cluster"].unique())
                        # Convert to strings for display; DBSCAN may return -1 for noise
                        cluster_options = [str(label) for label in unique_labels]
                        st.markdown("**選擇要顯示的 DBSCAN 群集** (可複選，預設顯示全部)")
                        selected_options = st.multiselect(
                            "Clusters to display", options=cluster_options, default=cluster_options
                        )
                        # Convert back to ints for filtering
                        selected_clusters = [int(opt) for opt in selected_options]
                        filtered_fail = die_shift_fail[die_shift_fail["DBSCAN_Cluster"].isin(selected_clusters)]

                        # Plot DBSCAN result for the selected clusters
                        st.subheader("DBSCAN Clustering (Fail die only)")
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(
                            data=filtered_fail,
                            x="Col",
                            y="Row",
                            hue="DBSCAN_Cluster",
                            palette="Set2",
                            s=100,
                            ax=ax2,
                        )
                        ax2.invert_yaxis()
                        st.pyplot(fig2)

                        # Merge DBSCAN clusters back to full die table on Row & Col
                        die_shift_clustering = die_shift_clustering.merge(
                            die_shift_fail[["Row", "Col", "DBSCAN_Cluster"]],
                            on=["Row", "Col"],
                            how="left",
                        )

            # Provide download of aggregated data with cluster labels (if any)
            csv = die_shift_clustering.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Clustered Data as CSV",
                csv,
                "clustered_die_data.csv",
                "text/csv",
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a probe mark Excel file and click '執行分析' to begin.")