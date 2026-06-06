# 🌫️ Air Quality CO Prediction — Multi Linear Regression

A **pure NumPy** implementation of multivariate linear regression trained on the [UCI Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality) to predict ground-truth **CO concentration (mg/m³)** from metal-oxide sensor readings and environmental conditions.

Training the model with gradient descent, math, and clean data.

---

## 📂 Dataset

**Source:** [UCI Machine Learning Repository — Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality)

- Recorded in an Italian city from **March 2004 to February 2005**
- Hourly averaged sensor responses from a multi-sensor device
- `CO(GT)` is the **target** — true CO concentration measured by a reference analyzer
- Missing values are encoded as **-200** and blank rows exist at the end of the file — both are cleaned before training

---

## ⚠️ Critical Data Cleaning Note

The dataset has **three types of bad data** that must all be handled:

| Problem | Count | Location |
|---|---|---|
| `-200` sentinel in target `CO(GT)` | 1,683 rows | Sensor offline |
| `-200` sentinel in **feature columns** | 330 rows | Sensor offline |
| Fully blank rows at end of CSV | 114 rows | Excel export artifact |

### 🧹 Cleaning Steps

```python
df.replace(-200, np.nan, inplace=True)
df.dropna(subset=['CO(GT)','PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH'], inplace=True)
df.dropna(how='all', inplace=True)
```

---

## 🧠 How It Works

```
Raw CSV  →  Replace -200 with NaN  →  Drop rows with any NaN
         →  Train/Test Split (80/20)  →  Standardize features
         →  Gradient Descent (10,000 steps)  →  Evaluate  →  Predict
```

**Gradient Descent Update Rules:**
```
ŷ  = X·m + b
dm = (-2/n) · Xᵀ·(y − ŷ)
db = (-2/n) · Σ(y − ŷ)

m  = m − lr·dm
b  = b − lr·db
```

---

## ⚙️ Features Used

| Feature | Description |
|---|---|
| `PT08.S1(CO)` | Tin oxide sensor — CO proxy |
| `PT08.S2(NMHC)` | Titania sensor — NMHC proxy |
| `PT08.S3(NOx)` | Tungsten oxide sensor — NOx proxy |
| `PT08.S4(NO2)` | Tungsten oxide sensor — NO2 proxy |
| `PT08.S5(O3)` | Indium oxide sensor — O3/NOx proxy |
| `T` | Temperature (°C) |
| `RH` | Relative Humidity (%) |
| `AH` | Absolute Humidity |

**Target:** `CO(GT)` — True hourly CO concentration in mg/m³

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **R² Score** | 0.8977 |
| **Train MSE** | 0.2410 |
| **Test MSE** | 0.1997 |

> Train and Test MSE are nearly identical — no overfitting. Clean rows used: **7,344** (from 9,471 raw).

---

## 🔍 Prediction Examples

### Example 1 — First Row of the Dataset

**Input:**
```
PT08.S1(CO)=1360  PT08.S2(NMHC)=1046  PT08.S3(NOx)=1056  PT08.S4(NO2)=1692
PT08.S5(O3)=1268  T=13.6  RH=48.9  AH=0.7578
```

**Run:**
```
Enter new data: 1360 1046 1056 1692 1268 13.6 48.9 0.7578
```

**Result:**
```
Predicted CO(GT) : 3.1541 mg/m³
Actual CO(GT)    : 2.6000 mg/m³
Error            : 0.5541 mg/m³
```

---

### Example 2 — Custom Sensor Reading

**Input:**
```
PT08.S1(CO)=1000  PT08.S2(NMHC)=900  PT08.S3(NOx)=1200  PT08.S4(NO2)=1400
PT08.S5(O3)=1100  T=25.0  RH=50.0  AH=0.8
```

**Run:**
```
Enter new data: 1000 900 1200 1400 1100 25.0 50.0 0.8
```

**Result:**
```
Predicted CO(GT) : 1.7408 mg/m³
```

> Falls near the dataset median (~1.8 mg/m³) — a typical mid-pollution reading.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/premchandsandra-sudo/Air-Quality-CO-Predictor.git
cd Air-Quality-CO-Predictor
```

### 2. Download the dataset
Get `AirQualityUCI.csv` from [UCI ML Repository](https://archive.ics.uci.edu/dataset/360/air+quality) and place it in the project root.

### 3. Install dependencies
```bash
pip install numpy pandas scikit-learn
```
> `scikit-learn` is used **only** for `train_test_split`. No model is imported from it.

### 4. Run
```bash
python air_quality.py
```

You'll be prompted:
```
Enter new data: 1360 1046 1056 1692 1268 13.6 48.9 0.7578
```

Enter **8 space-separated values** in this exact order:
```
PT08.S1(CO)  PT08.S2(NMHC)  PT08.S3(NOx)  PT08.S4(NO2)  PT08.S5(O3)  T  RH  AH
```

---

## 📁 Project Structure

```
Air-Quality-CO-predictor/
├── air_quality.py        # Main script — training + prediction
├── AirQualityUCI.csv     # Dataset (download separately from UCI)
└── README.md
```

---

## 📜 License

MIT License. Dataset belongs to UCI ML Repository — cite accordingly if used in research.

---

## 🔗 Reference

> S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia,
> *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario*,
> Sensors and Actuators B: Chemical, 2008.
> UCI ML Repository: https://archive.ics.uci.edu/dataset/360/air+quality
