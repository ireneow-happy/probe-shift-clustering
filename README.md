# 🔍 Probe Mark Shift Clustering App

本專案是一個用於晶圓探針偏移量分析的視覺化工具，基於 [Streamlit](https://streamlit.io/) 架構開發。使用者可上傳探針資料檔案，自行選擇分析方向與統計方式，並使用 KMeans 或 DBSCAN 模型進行聚類分析與視覺化呈現。

---

## 📦 功能特色

- 計算探針偏移量：
  - 垂直偏移 = |Prox Up - Prox Down|
  - 水平偏移 = |Prox Left - Prox Right|
- 熱力圖呈現偏移分布（支援最大值與平均值）
- 支援兩種聚類模型：KMeans 與 DBSCAN
- 自動產生 K-distance plot（輔助選擇 DBSCAN 的 eps 參數）
- 匯出聚類結果為 CSV 檔案

---

## 🚀 使用說明

### 1. 安裝相依套件

請使用 Python 3.8+，建議建立虛擬環境並安裝：

```bash
pip install -r requirements.txt
```

### 2. 執行程式

```bash
streamlit run app.py
```

### 3. 上傳檔案格式

請上傳一個包含以下欄位的 `.xlsx` Excel 檔案：

| Row | Col | Prox Up | Prox Down | Prox Left | Prox Right |
|-----|-----|---------|-----------|-----------|------------|

---

## 🖼 畫面說明

- 📊 偏移量熱力圖（垂直或水平）
- 🧩 KMeans 聚類圖
- 🧩 DBSCAN 聚類圖
- 📐 DBSCAN K-distance plot
- 📥 下載含聚類資訊的 CSV

---

## 📁 requirements.txt

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
kneed
```

---

## 🛠 技術架構

- 前端：Streamlit
- 分析套件：scikit-learn, seaborn, matplotlib
- 資料處理：pandas, numpy

---

## 📬 聯絡方式

若有任何建議、錯誤回報或改進想法，歡迎透過 Issue 或 Pull Request 聯絡我。
