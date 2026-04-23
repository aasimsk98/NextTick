# NextTick
### Predicting Stock Market Direction and Magnitude Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)

---

## Overview

**NextTick** is an end-to-end machine learning system that predicts next-day stock market movement using historical price data and engineered financial indicators.

The system addresses two predictive tasks in parallel:
- **Classification** - Predict whether a stock will go **UP or DOWN** tomorrow
- **Regression** - Predict the **percentage magnitude** of that price change

Trained on 50 stocks across 8 sectors of the S&P 500, the deployed Flask app accepts any ticker a user inputs, fetches live market data via `yfinance` at inference time, computes features on the fly, and returns a real-time prediction.

---

## Motivation

Stock market prediction is one of the most studied problems in financial data science. Institutions like Fidelity Investments rely on quantitative research and ML-driven signals to support portfolio management, risk analytics, and investment strategy. NextTick is a practical, deployable mini-version of exactly that - a demonstration of the full ML lifecycle from raw data to live prediction through a web interface.

---

## Project Architecture

```
User types ticker (e.g. "AAPL") on Flask webpage
                    |
                    v
       Flask backend receives the request
                    |
                    v
       App fetches latest stock data via yfinance
                    |
                    v
       21 features computed (technical + market + OHLCV)
                    |
                    v
       Trained models loaded from disk (.pkl / .pt)
                    |
                    v
       Inference -> direction (UP/DOWN) + magnitude (% change)
                    |
                    v
       Result displayed on frontend
```

---

## Model Stack

Each task is served by three models of increasing sophistication, giving us a linear baseline, a nonlinear ensemble, and a deep sequence model.

### Classification (Direction - UP / DOWN)

| Model | Type | Framework |
|---|---|---|
| Logistic Regression | Linear baseline | scikit-learn |
| Random Forest Classifier | Tuned tree ensemble | scikit-learn |
| LSTM Classifier | Stacked recurrent network | PyTorch |

### Regression (Magnitude - % change)

| Model | Type | Framework |
|---|---|---|
| Linear Regression | OLS baseline | scikit-learn |
| Random Forest Regressor | Tuned tree ensemble | scikit-learn |
| LSTM Regressor | Stacked recurrent network | PyTorch |

---

## Dataset

| Property | Details |
|---|---|
| Source | `yfinance` (Yahoo Finance) |
| Time Span | 5 years of daily data (2021-05-18 to 2026-04-17) |
| Tickers | 50 stocks across 8 sectors |
| Total Rows | 61,750 |
| Rows per Ticker | 1,235 (uniform after warmup / target alignment) |
| Features | 21 |

### Universe (50 tickers, 8 sectors)

| Sector | ETF | Tickers |
|---|---|---|
| Technology | XLK | AAPL, MSFT, GOOGL, NVDA, CRM, ORCL, ADBE |
| Consumer Discretionary | XLY | AMZN, TSLA, HD, NKE, MCD, SBUX |
| Financials | XLF | JPM, GS, BAC, V, MA, BLK |
| Healthcare | XLV | JNJ, PFE, UNH, LLY, ABBV, MRK, TMO |
| Energy | XLE | XOM, CVX, COP, SLB, EOG |
| Industrials | XLI | CAT, BA, HON, UPS, DE, GE |
| Consumer Staples | XLP | PG, KO, WMT, COST, PEP, PM |
| Communication Services | XLC | META, NFLX, DIS, VZ, T, CMCSA, TMUS |

---

## Features (21 total)

### Price / Technical (6)
| Feature | Description |
|---|---|
| `daily_return` | Percent change from previous close |
| `sma_10`, `sma_20` | 10 and 20-day simple moving averages |
| `volatility_10` | 10-day rolling standard deviation of returns |
| `momentum_10` | 10-day price change |
| `rsi_14` | 14-day Relative Strength Index |

### Market / Macro (10)
| Feature | Description |
|---|---|
| `spy_return` | Daily return of the broad US equity market (SPY) |
| `vix_level` | Level of the VIX volatility index |
| `sector_return` | Daily return of this stock's sector ETF |
| `relative_to_spy` | Stock return minus SPY return |
| `relative_to_sector` | Stock return minus sector-ETF return |
| `tnx_change` | Daily change in the 10-year Treasury yield |
| `dxy_change` | Daily percent change in the US dollar index |
| `oil_return` | Daily return of USO (oil proxy) |
| `day_of_week`, `month` | Calendar features |

### OHLCV-Derived (5)
| Feature | Description |
|---|---|
| `overnight_gap` | Today's open vs yesterday's close |
| `intraday_return` | Close-vs-open within the same session |
| `daily_range_pct` | (High - Low) / Close |
| `close_location` | Where close sits inside the day's range, in [0, 1] |
| `relative_volume` | Today's volume vs its 20-day mean |

### Targets
- **`target_direction`** - 1 if tomorrow's close > today's close, else 0
- **`target_return`** - percent change between today's close and tomorrow's close

---

## Methodology

### Data Split
Chronological **70/15/15** train/validation/test. Chronological because this is time-series data - training on earlier dates and testing on later ones prevents information leaking from future into past.

### Preprocessing
A `StandardScaler` fit only on training data is applied uniformly to validation and test. Random Forest doesn't strictly need scaling, but applying it uniformly keeps preprocessing consistent across all six models.

### Hyperparameter Tuning
Random Forest uses `GridSearchCV` with `TimeSeriesSplit(5)` - expanding-window cross-validation that respects time order, unlike standard k-fold which would shuffle randomly and leak future data.

| Hyperparameter | Grid |
|---|---|
| `n_estimators` | 100, 200, 300, 400, 500 |
| `max_depth` | 5, 10, 20 |
| `min_samples_leaf` | 5, 10 |

Logistic Regression and Linear Regression use defaults (no tuning).

The LSTM uses a fixed architecture (see below), tuned by extensive prior experimentation.

### LSTM Architecture

```
Input (batch, 30 timesteps, 21 features)
  -> LSTM(64, return_sequences=True)
  -> Dropout(0.2)
  -> LSTM(32, final hidden state only)
  -> Dropout(0.2)
  -> Dense(32 -> 16) + ReLU
  -> Dense(16 -> 1) + [sigmoid for classification / linear for regression]
```

Training: Adam optimizer, learning rate 1e-3, batch size 64, up to 30 epochs with early-stopping patience 5 on validation loss. Classification uses BCE loss; regression uses MAE (L1) loss.

---

## Evaluation Metrics

| Task | Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, Macro F1, Confusion Matrix |
| Regression | MAE (Mean Absolute Error), RMSE, R-squared |

---

## Results (Test Set)

### Classification

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|
| Logistic Regression | 0.5068 | 0.5097 | 0.8917 | 0.4107 |
| Random Forest | 0.4991 | 0.5075 | 0.6380 | 0.4873 |
| **LSTM** | **0.5139** | 0.5171 | 0.6970 | **0.4944** |

Best classifier: **LSTM** (by Macro F1).

### Regression

| Model | MAE | RMSE | R-squared |
|---|---|---|---|
| **Linear Regression** | **0.012728** | 0.018346 | -0.0019 |
| Random Forest | 0.012789 | 0.018404 | -0.0082 |
| LSTM | 0.013017 | 0.018639 | +0.0019 |

Best regressor: **Linear Regression** (by MAE, narrowly).

---

## Deployment

The trained models are loaded by a Flask web app that takes a user-entered ticker, fetches recent data via `yfinance`, computes the 21 features, and runs inference.

| Component | Details |
|---|---|
| Web framework | Flask |
| Model loading | `pickle` for sklearn models, `torch.load` for PyTorch |
| Data fetching | `yfinance` at inference time |
| Preprocessing | Shared `StandardScaler` (same one used during training) |

---

## Repository Structure

```
NextTick/
|
├── data/
|   └── processed/
|       └── nexttick_dataset_50tickers.csv      <- Final dataset (30.2 MB)
|
├── notebooks/
|   ├── 01_data_pipeline.ipynb                  <- Fetch, clean, engineer features
|   ├── 02_classification_models.ipynb          <- LogReg + RF + LSTM classifiers
|   └── 03_regression_models.ipynb              <- LinReg + RF + LSTM regressors
|
├── models/
|   ├── logistic_regression.pkl
|   ├── random_forest_classifier.pkl
|   ├── lstm_classifier.pt
|   ├── linear_regression.pkl
|   ├── random_forest_regressor.pkl
|   ├── lstm_regressor.pt
|   └── scaler.pkl                              <- Shared StandardScaler
|
├── app/                                         <- Flask web application
|   └── ...
|
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Data | `yfinance`, `pandas`, `numpy`, `pandas_ta` |
| ML | `scikit-learn`, `PyTorch` |
| Visualization | `matplotlib`, `seaborn` |
| Web App | `Flask` |
| Version Control | `Git`, `GitHub` |

---

## Build Phases

| # | Phase | Goal |
|---|---|---|
| 1 | Data Pipeline | Fetch 50 tickers, engineer 21 features, generate targets, save CSV |
| 2 | Classification Models | Train and compare LogReg, RF Classifier, LSTM Classifier |
| 3 | Regression Models | Train and compare LinReg, RF Regressor, LSTM Regressor |

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/NextTick.git
cd NextTick

# Install dependencies
pip install -r requirements.txt

# Run the notebooks in order
jupyter notebook notebooks/01_data_pipeline.ipynb
jupyter notebook notebooks/02_classification_models.ipynb
jupyter notebook notebooks/03_regression_models.ipynb

# Run the Flask app
cd app
flask run
```

---