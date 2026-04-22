# Household Energy Predictor

Predicting household energy consumption using machine learning | Built as a learning project in data science

---

## Project Overview

This project builds a machine learning model to predict household electricity consumption (in kilowatts) using historical minute-by-minute power readings. The goal is to understand the patterns behind energy usage and produce accurate predictions from measurable electrical features.

This is an end-to-end data science project covering exploratory data analysis, data cleaning, preprocessing, model training, and evaluation.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

**Description:** Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. The dataset contains over 2 million observations.

**To use this project:**
1. Download the dataset from the link above
2. Place `household_power_consumption.txt` in the `data/raw/` directory

### Features

| Feature | Description |
|---|---|
| `Global_active_power` | Household global minute-averaged active power (kW) — **target variable** |
| `Global_reactive_power` | Household global minute-averaged reactive power (kW) |
| `Voltage` | Minute-averaged voltage (V) |
| `Global_intensity` | Household global minute-averaged current intensity (A) |
| `Sub_metering_1` | Energy sub-metering for kitchen appliances (Wh) |
| `Sub_metering_2` | Energy sub-metering for laundry appliances (Wh) |
| `Sub_metering_3` | Energy sub-metering for climate control appliances (Wh) |

---

## Project Structure

```
household-energy-predictor/
│
├── data/
│   ├── raw/                        ← place downloaded dataset here
│   └── processed/                  ← cleaned and preprocessed data
│
├── notebooks/
│   ├── eda.ipynb                   ← exploratory data analysis
│   ├── preprocessing.ipynb         ← data cleaning and feature engineering
│   └── model_train.ipynb           ← model training and evaluation
│
├── requirements.txt                ← Python dependencies
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis
- Investigated data shape, types, and missing values
- Found that `?` was used as a placeholder for missing values across all columns
- Identified 1.25% of rows fully corrupted — all features missing simultaneously
- Detected right-skewed distribution in the target variable with genuine high-consumption outliers
- Correlation analysis revealed `Global_intensity` had a perfect 1.0 correlation with the target — flagged as redundant
- `Sub_metering_3` showed the strongest relationship with energy consumption (0.64)

### 2. Data Cleaning and Preprocessing
- Replaced `?` placeholders with `NaN` and converted all numeric columns from string to float
- Dropped 1.25% of fully corrupt rows
- Merged `Date` and `Time` columns into a single `Datetime` column
- Extracted time features: `Year`, `Month`, `Day`, `Hour`, `Minute`
- Dropped `Global_intensity` to eliminate data leakage

### 3. Model Training
Trained and compared two models:

- **Random Forest Regressor** — baseline model, strong out of the box but slow on large datasets
- **XGBoost Regressor** — final model, significantly faster and more scalable, improved further with hyperparameter tuning

---

## Results

**Final Model: XGBoost Regressor (hyperparameter tuned)**

| Metric | Value | Interpretation |
|---|---|---|
| RMSE | 0.3426 | Predictions are off by ~0.34 kW on average |
| R² | 89.42% | Model explains 89.4% of variance in energy consumption |
| MAPE | 27.32% | Average percentage error across all predictions |

### Hyperparameters used
```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

---

## Key Findings

- Household energy consumption is heavily concentrated below 2 kW, with occasional spikes up to 11 kW
- Sub_metering_3 (climate control) is the strongest appliance-level predictor of total consumption
- XGBoost trained on the full 2 million row dataset in under 35 seconds — significantly more practical than Random Forest for large scale tabular data
- Dropping the perfectly correlated `Global_intensity` column was critical — keeping it inflated R² to 99% through data leakage

---

## Future Improvements

- Add lag features (previous 1 min, 1 hour, 1 day consumption)
- Add rolling statistics (rolling mean and std over sliding windows)
- Apply cyclical encoding to hour, day, and month features
- Explore LSTM neural networks for sequence-based time series forecasting
- Train final model on complete 2 million row dataset

---

## Setup and Usage

### 1. Clone the repository
```bash
git clone https://github.com/BilaalBakare/household-energy-predictor.git
cd household-energy-predictor
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from the [UCI repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) and place in `data/raw/`

### 5. Run the notebooks in order
```
notebooks/eda.ipynb
notebooks/preprocessing.ipynb
notebooks/model_train.ipynb
```

---

## Tech Stack

- **Python 3**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **matplotlib / seaborn** — data visualisation
- **scikit-learn** — model selection, evaluation, Random Forest
- **XGBoost** — gradient boosting model

---

## Author

**Bilaal Bakare**  
