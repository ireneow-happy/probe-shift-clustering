
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

st.title("Probe Mark Shift Clustering App")

st.markdown("""
**📘 計算邏輯與功能說明**

- 偏移量（偏心度）計算公式：
  - 垂直偏移 = |Prox Up - Prox Down|
  - 水平偏移 = |Prox Left - Prox Right|
  - 總偏移 = 垂直偏移 + 水平偏移

- 本工具可支援：
  - Total Shift 偏移量熱力圖
  - KMeans 與 DBSCAN 分群比較
  - DBSCAN 自動分析 eps（K-distance plot）
  - 下載含分群資訊的結果檔
""")

# Sidebar for user inputs
st.sidebar.header("Settings")
model_selection = st.sidebar.multiselect(
    "Select Clustering Methods",
    ["KMeans", "DBSCAN"],
    default=["KMeans"]
)

if "KMeans" in model_selection:
    k_value = st.sidebar.number_input("KMeans: Number of Clusters (K)", min_value=2, max_value=20, value=3, step=1)
if "DBSCAN" in model_selection:
    eps_value = st.sidebar.slider("DBSCAN: eps (neighborhood size)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    min_samples = st.sidebar.slider("DBSCAN: min_samples", min_value=2, max_value=20, value=5, step=1)

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
run_analysis = st.button("🚀 執行分析")

if uploaded_file is not None and run_analysis:
    try:
        df = pd.read_excel(uploaded_file)

        # Calculate shift values
        df["Vertical Shift"] = np.abs(df["Prox Up"] - df["Prox Down"])
        df["Horizontal Shift"] = np.abs(df["Prox Left"] - df["Prox Right"])
        df["Total Shift"] = df["Vertical Shift"] + df["Horizontal Shift"]

        # Group by die position and calculate mean shift
        die_shift = df.groupby(["Row", "Col"])["Total Shift"].mean().reset_index()

        # Heatmap visualization
        st.subheader("Heatmap of Total Shift")
        heatmap_data = die_shift.pivot(index="Row", columns="Col", values="Total Shift")
        fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax_hm)
        ax_hm.invert_yaxis()
        ax_hm.set_title("Total Shift Heatmap")
        st.pyplot(fig_hm)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(die_shift[["Col", "Row", "Total Shift"]])

        # K-distance plot for DBSCAN
        if "DBSCAN" in model_selection:
            st.subheader("📐 K-distance Plot for DBSCAN (use to determine eps)")
            nbrs = NearestNeighbors(n_neighbors=min_samples).fit(features_scaled)
            distances, _ = nbrs.kneighbors(features_scaled)
            k_distances = np.sort(distances[:, -1])

            knee_locator = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
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

        if "KMeans" in model_selection:
            kmeans = KMeans(n_clusters=k_value, random_state=42)
            clusters_kmeans = kmeans.fit_predict(features_scaled)
            die_shift["KMeans_Cluster"] = clusters_kmeans

            st.subheader("KMeans Clustering")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=die_shift, x="Col", y="Row", hue="KMeans_Cluster", palette="Set2", s=100, ax=ax1)
            ax1.invert_yaxis()
            ax1.set_title(f"KMeans Clustering (K={k_value})")
            st.pyplot(fig1)

        if "DBSCAN" in model_selection:
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
            clusters_dbscan = dbscan.fit_predict(features_scaled)
            die_shift["DBSCAN_Cluster"] = clusters_dbscan

            st.subheader("DBSCAN Clustering")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=die_shift, x="Col", y="Row", hue="DBSCAN_Cluster", palette="Set2", s=100, ax=ax2)
            ax2.invert_yaxis()
            ax2.set_title(f"DBSCAN Clustering (eps={eps_value}, min_samples={min_samples})")
            st.pyplot(fig2)

        csv = die_shift.to_csv(index=False).encode("utf-8")
        st.download_button("Download Clustered Data as CSV", csv, "clustered_die_data.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a probe mark Excel file and click '執行分析' to begin.")
