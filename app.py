
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("Probe Mark Shift Clustering App")

# Sidebar for user inputs
st.sidebar.header("Settings")
k_value = st.sidebar.number_input("Number of Clusters (K)", min_value=2, max_value=20, value=3, step=1)
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        # Calculate shift values
        df["Vertical Shift"] = np.abs(df["Prox Up"] - df["Prox Down"])
        df["Horizontal Shift"] = np.abs(df["Prox Left"] - df["Prox Right"])
        df["Total Shift"] = df["Vertical Shift"] + df["Horizontal Shift"]

        # Group by die position and calculate mean shift
        die_shift = df.groupby(["Row", "Col"])["Total Shift"].mean().reset_index()

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(die_shift[["Col", "Row", "Total Shift"]])

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k_value, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        die_shift["Cluster"] = clusters

        # Plot the result
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=die_shift, x="Col", y="Row", hue="Cluster", palette="Set2", s=100)
        plt.title(f"Clustering of Dies (K={k_value})")
        plt.gca().invert_yaxis()
        plt.xlabel("Col")
        plt.ylabel("Row")
        plt.legend(title="Cluster")
        st.pyplot(plt)

        # Allow download of cluster data
        csv = die_shift.to_csv(index=False).encode("utf-8")
        st.download_button("Download Clustered Data as CSV", csv, "clustered_die_data.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a probe mark Excel file to begin.")
