# NextTick - Next-Day Stock Market Forecasting

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)
![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20S3-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

---

## Live Demo

**[http://nexttick.duckdns.org](http://nexttick.duckdns.org)**

Hosted on AWS (EC2 + S3 + IAM, containerized with Docker). See [DEPLOY.md](DEPLOY.md) for the full deployment walkthrough.

---

## Overview

**NextTick** is an end-to-end machine learning system that predicts next-day stock market movement using historical price data and engineered financial indicators.

The system addresses two predictive tasks in parallel:
- **Classification** - Predict whether a stock will go **UP or DOWN** tomorrow
- **Regression** - Predict the **percentage magnitude** of that price change

Trained on 50 stocks across 8 sectors of the S&P 500, the project delivers six trained models (three classifiers and three regressors) served through a Flask web application where a user can enter any ticker and receive a live forecast.

---

## Motivation

Stock market prediction is one of the most studied problems in financial data science. NextTick is a practical demonstration of the full ML lifecycle - from raw data ingestion through feature engineering, model training, and evaluation - culminating in a web application that serves live predictions to any user.

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

## Inference Flow (How the Flask App Works)

Training is offline and frozen. Inference happens live on every user request. The models' weights are saved once (after running the notebooks) and never modified again - every prediction reuses the same trained artifacts.

When a user types a ticker (e.g. AAPL) and clicks **Run Forecast**:

1. **Fetch** the last ~6 months of OHLCV data for the ticker from Yahoo Finance
2. **Fetch** market context (SPY, VIX, 10Y Treasury yield, dollar index, oil, and 8 sector ETFs) to compute market/macro features
3. **Engineer** the 21 features on the fetched data. Each ticker is mapped to its sector ETF (XLK for tech, XLF for financials, etc). Tickers outside the trained 50 fall back to SPY as the sector benchmark.
4. **Scale** features using the saved `StandardScaler` - the exact same one fit during training
5. **Run inference**:
    - Logistic Regression and Random Forest take the **most recent row** (1 x 21 vector) and output P(Up) or predicted return
    - LSTM takes the **most recent 30 rows** as a sequence (1 x 30 x 21 tensor) and outputs P(Up) or predicted return
6. **Aggregate** the six predictions:
    - **Direction:** `"Up"` if the mean of the three classifier probabilities is >= 0.5, else `"Down"`
    - **Confidence:** `|mean_prob - 0.5| * 2`, scaled to 0-100%
    - **Magnitude:** arithmetic mean of the three regressor outputs (in percent points)
    - **Projected close:** `last_close * (1 + magnitude / 100)`
7. **Return** all numbers to the UI, which renders the headline direction/magnitude cards, a 30-day history chart, the per-model breakdown, and a step-by-step analysis walkthrough

Fetching more than 6 months would not change the prediction - only the last 30 rows are ever fed to the LSTM, and only the last row is fed to LogReg / Random Forest. The 6-month default gives a comfortable buffer for rolling-window warmup and the history chart.

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
├── flask_app/                                   <- Flask web application
|   ├── app.py                                   <- Flask routes and orchestration
|   ├── Dockerfile                               <- Container definition (CPU-only torch)
|   ├── .dockerignore
|   ├── requirements.txt                         <- Local dev dependencies
|   ├── requirements-docker.txt                  <- Container dependencies (no torch)
|   ├── utils/
|   |   ├── features.py                          <- 21-feature engineering
|   |   ├── fetcher.py                           <- Yahoo Finance data fetch + market context
|   |   ├── inference.py                         <- Model loading and prediction
|   |   └── s3_loader.py                         <- Boto3-based artifact loader (deploy mode)
|   ├── models/                                  <- Trained artifacts dropped here (local mode)
|   ├── static/
|   |   ├── css/style.css
|   |   ├── js/app.js
|   |   └── sample_input.csv
|   └── templates/
|       ├── base.html
|       └── index.html
|
├── requirements.txt
├── README.md
└── DEPLOY.md                                    <- AWS deployment walkthrough
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Data | `yfinance`, `pandas`, `numpy`, `pandas_ta` |
| ML | `scikit-learn`, `PyTorch` |
| Visualization | `matplotlib`, `seaborn` |
| Web App | `Flask`, `gunicorn` |
| Container | `Docker` |
| Cloud | `AWS EC2`, `AWS S3`, `AWS IAM`, `boto3` |
| Version Control | `Git`, `GitHub` |

---

## Build Phases

| # | Phase | Goal |
|---|---|---|
| 1 | Data Pipeline | Fetch 50 tickers, engineer 21 features, generate targets, save CSV |
| 2 | Classification Models | Train and compare LogReg, RF Classifier, LSTM Classifier |
| 3 | Regression Models | Train and compare LinReg, RF Regressor, LSTM Regressor |
| 4 | Cloud Deployment | Containerize with Docker, deploy to AWS EC2 with S3-backed model artifacts |

---

## Installation & Usage

### Option 1 - Run locally (development)

#### Train the models

```bash
# Clone the repository
git clone https://github.com/aasimsk98/NextTick.git
cd NextTick

# Install dependencies
pip install -r requirements.txt

# Run the notebooks in order
jupyter notebook notebooks/01_data_pipeline.ipynb
jupyter notebook notebooks/02_classification_models.ipynb
jupyter notebook notebooks/03_regression_models.ipynb
```

Each notebook saves its trained model artifacts to `models/`.

#### Run the Flask app locally

The Flask app loads the six trained models and serves predictions on a web UI. From the repository root:

```bash
# Step 1: Move into the Flask app directory
cd flask_app

# Step 2: Create and activate a virtual environment
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Step 3: Install Flask app dependencies
pip install -r requirements.txt

# Step 4: Launch the app
python app.py
```

The app will start on `http://127.0.0.1:5000`. Open it in a browser, search for a stock ticker (e.g. AAPL, TSLA, MSFT), and click **Run Forecast**.

The `flask_app/models/` directory should contain the trained artifacts (`logistic_regression.pkl`, `random_forest_classifier.pkl`, `lstm_classifier.pt`, `linear_regression.pkl`, `random_forest_regressor.pkl`, `lstm_regressor.pt`, `scaler.pkl`) produced by the notebooks. If you retrain the models, overwrite these files to use the new versions.

#### Run with Docker locally (optional)

```bash
cd flask_app
docker build -t nexttick:local .

# With local model files mounted
docker run --rm -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  nexttick:local
```

Visit `http://localhost:5000`.

### Option 2 - Deploy to AWS

See **[DEPLOY.md](DEPLOY.md)** for the complete AWS deployment walkthrough (EC2 + S3 + IAM + Docker). Stays within AWS Free Tier.

---