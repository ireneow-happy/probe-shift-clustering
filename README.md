# Probe Mark Analysis App

本應用提供三種分析模式，協助使用者進行 Probe Mark 資料分析：

---

## 🔍 可選擇的分析類型

### 1. Clustering analysis
使用 DBSCAN 和 KMeans 進行失敗 die 的群集分析。可視化顯示各群集於 wafer 上的分布。

- 可設定參數：
  - 偏移方向（Vertical / Horizontal）
  - 統計方式（max / average）
  - 使用者可自選 clustering 方法（DBSCAN、KMeans）
  - DBSCAN: `eps` 和 `min_samples`
  - KMeans: 群集數 `K`
- 可用 checkbox 選擇要顯示哪些群集

---

### 2. DUT analysis
針對 Probe Card 上的 DUT# 分析其失敗率與分布情形。

- 分析項目：
  - 計算每個 DUT# 的失敗率
  - 依照 DUT# 繪製 fail 分布地圖
  - 進行統計檢定以確認是否有顯著差異

---

### 3. Trend analysis
針對測試順序（TD Order）進行時間序列趨勢分析。

- 以 shift 方向（Up / Down / Left / Right）觀察 probe mark 是否有偏移趨勢
- 可視化 shift vs. TD Order 的關係圖
- 檢查是否出現系統性偏移

---

## ⚙️ 操作說明

1. 左側選擇分析模式後，會顯示該模式對應的參數設定區。
2. 設定參數後，**必須點擊「執行群集分析」按鈕**，才會真正進行運算與繪圖。
3. 若使用互動式版本，勾選或取消群集會觸發頁面 rerun，但分析結果會儲存在 `session_state` 中，不會重新計算。
4. Trend 分析模式中可使用 K-distance 曲線來協助設定 DBSCAN `eps` 值。
5. 所有圖表都會以 Fail die 為主體進行分析與過濾。

---

如有需要，可依需求擴充支援更多欄位與分析邏輯。