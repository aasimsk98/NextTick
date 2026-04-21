# NextTick
### Predicting Stock Market Direction and Magnitude Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-yellow)

---

## Overview

**NextTick** is an end-to-end machine learning system that predicts next-day stock market movement using historical price data and engineered financial indicators.

The system addresses two predictive tasks simultaneously:
- **Classification** — Predict whether a stock will go **UP or DOWN** tomorrow
- **Regression** — Predict the **percentage magnitude** of that price change

Trained on 10 diverse stocks across tech and finance sectors, the deployed app generalizes to **any ticker** a user types — fetching live market data via `yfinance` at inference time and returning a real-time prediction with a confidence score.

---

## Motivation

Stock market prediction is one of the most studied problems in financial data science. Institutions like Fidelity Investments rely on quantitative research and ML-driven signals to support portfolio management, risk analytics, and investment strategy. NextTick is a mini version of exactly that - a practical, deployable system that demonstrates the full ML lifecycle from raw data to live prediction.

---

## Project Architecture

```
User types ticker (e.g. "AAPL") on webpage
            ↓
Request sent to FastAPI backend (hosted on AWS EC2)
            ↓
App fetches latest stock data via yfinance
            ↓
Features engineered (SMA, RSI, Momentum, Volatility)
            ↓
Trained model loaded from AWS S3
            ↓
Inference → "UP — 63% confident, +1.2% magnitude"
            ↓
Result displayed on frontend
```

---

## Model Stack

### Classification (Direction - UP / DOWN)
| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest Classifier | Robust ensemble |
| LSTM (PyTorch) | Deep learning - captures temporal sequences |

### Regression (Magnitude — % change)
| Model | Type |
|---|---|
| Linear Regression | Linear baseline |
| XGBoost Regressor | Sequential boosting - precise for continuous values |
| LSTM (PyTorch) | Deep learning - same architecture, continuous output |

---

## Dataset

| Property | Details |
|---|---|
| Source | `yfinance` (Yahoo Finance) |
| Time Span | 5 years of daily data |
| Tickers | 10 stocks across tech + finance sectors |
| Total Rows | ~12,500 |
| Trading Days/Year | ~252 |
| Classes | 2 (UP, DOWN) |
| Samples per Class | >500 |

**Example Tickers:**
- Tech: `AAPL`, `MSFT`, `TSLA`, `GOOGL`, `AMZN`
- Finance: `JPM`, `GS`, `BAC`

---

## Features

### Raw Features (from yfinance)
- Open, High, Low, Close, Volume (OHLCV)

### Engineered Features (via `pandas_ta`)
| Feature | Description |
|---|---|
| SMA (7, 20, 50 day) | Simple Moving Average - smooths price noise |
| Volatility | Rolling std of daily returns - measures price swings |
| Momentum | Rate of price change over N days |
| RSI (0–100) | Relative Strength Index - overbought/oversold signal |
| Daily Return | % change from previous day's close |

### Labels
- **Classification:** UP (1) if tomorrow's close > today's close, DOWN (0) otherwise
- **Regression:** % change between today's close and tomorrow's close

---

## Evaluation Metrics

| Task | Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, F1-Score |
| Regression | MAE (Mean Absolute Error), RMSE (Root Mean Squared Error) |

> **Note:** Time-series cross-validation (`TimeSeriesSplit`) is used throughout - never standard k-fold - to preserve chronological order and prevent data leakage.

---

## Deployment (AWS)

| Service | Purpose |
|---|---|
| AWS S3 | Store raw data CSVs and trained model artifacts |
| AWS EC2 | Host FastAPI prediction API (always live) |
| Frontend | Streamlit or HTML webpage for user interaction |

---

## Repository Structure

```
NextTick/
│
├── data/
│   ├── raw/                    <- Raw stock CSVs downloaded from yfinance
│   └── processed/              <- Cleaned, feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb  <- Fetch, clean, engineer features
│   ├── 02_baseline_models.ipynb<- Logistic + Linear Regression
│   ├── 03_ensemble_models.ipynb<- Random Forest + XGBoost
│   ├── 04_lstm_model.ipynb     <- LSTM in PyTorch (classification + regression)
│   └── 05_comparison.ipynb     <- Results, visualizations, analysis
│
├── src/
│   ├── data_pipeline.py        <- Data fetching and cleaning
│   ├── features.py             <- Feature engineering functions
│   ├── models.py               <- Model training and saving
│   └── evaluate.py             <- Evaluation metrics and plots
│
├── api/
│   └── main.py                 <- FastAPI app (inference endpoint)
│
├── requirements.txt            <- All dependencies
└── README.md                   <- You are here
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Data | `yfinance`, `pandas`, `numpy`, `pandas_ta` |
| ML | `scikit-learn`, `xgboost`, `PyTorch` |
| Visualization | `matplotlib`, `seaborn` |
| Deployment | `FastAPI`, `AWS EC2`, `AWS S3` |
| Frontend | `Streamlit` |
| Version Control | `Git`, `GitHub` |

---

## Build Roadmap

| No | Phase | Goal |
|---|---|---|
| 1 | Data Pipeline | Fetch, clean, engineer features, save processed CSV |
| 2 | Baseline Models | Logistic Regression + Linear Regression |
| 3 | Ensemble Models | Random Forest + XGBoost + TimeSeriesSplit tuning |
| 4 | LSTM (PyTorch) | Deep learning models for both tasks |
| 5 | Comparison & Analysis | Results table, visualizations, insights |
| 6 | AWS Deployment | FastAPI + EC2 + S3 + live frontend |

---

## Scaling Plan

| Phase | Tickers | Status |
|---|---|---|
| Development | 10 | Start here |
| Final Training | 20–50 | Scale up if feasible |
| Stretch Goal | 100+ | Time permitting |
| Ambitious Stretch | 500+ | If compute allows |

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/NextTick.git
cd NextTick

# Install dependencies
pip install -r requirements.txt

```

---
