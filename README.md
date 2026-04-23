# NextTick — Next-Day Stock Market Forecasting

**CS 6140 · Machine Learning · Prof. Ehsan Elhamifar**
**Team:** Pratham Pradeep Mahajan · Mohammad Aasim Shaikh

A dual-engine machine-learning system that isolates the **direction** of the
next trading day's move from its **magnitude**, trained on five years of
cross-sector daily data pulled via `yfinance`.

---

## Repository layout

```
NextTick/
├── notebooks/
│   └── NextTick_Training.ipynb      # Colab-ready training pipeline
└── flask_app/
    ├── app.py                       # Flask inference server
    ├── requirements.txt
    ├── utils/
    │   ├── features.py              # Shared feature engineering (mirror of notebook)
    │   └── inference.py             # Loads models, runs predictions
    ├── models/                      # ← drop trained artifacts here
    ├── static/
    │   ├── css/style.css
    │   ├── js/app.js
    │   └── sample_input.csv         # 30-day sample OHLCV for quick demo
    └── templates/
        ├── base.html
        └── index.html
```

---

## Part 1 — Train the models (Colab)

1. Open `notebooks/NextTick_Training.ipynb` in [Google Colab](https://colab.research.google.com/).
2. `Runtime → Change runtime type → T4 GPU` (LSTM trains ~4× faster than on CPU).
3. `Runtime → Run all`.
4. The final cell downloads `nexttick_artifacts.zip` containing eight files:
   - Four sklearn models as `.pkl`
   - Two Keras LSTM models as `.keras`
   - `scaler.pkl`
   - `metadata.json`

**Why `.keras` for LSTM instead of `.pkl`?** Keras models hold TensorFlow graph
state that does not pickle reliably across TF versions. The Flask app loads
sklearn models through `joblib.load` and Keras models through
`tensorflow.keras.models.load_model`.

---

## Part 2 — Run the Flask app locally

```bash
cd flask_app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Unzip `nexttick_artifacts.zip` into `flask_app/models/` so the folder contains
all eight files, then:

```bash
python app.py
```

Open <http://localhost:5000>, drop a CSV onto the upload zone, and hit
**Run forecast**.

---

## Input CSV format

Minimum 30 consecutive trading days. Required columns (case-insensitive):

| Date       | Open   | High   | Low    | Close  | Volume      |
|------------|--------|--------|--------|--------|-------------|
| 2026-03-03 | 185.81 | 190.39 | 184.49 | 189.31 | 61,473,377  |
| …          | …      | …      | …      | …      | …           |

A ready-to-use example lives at `flask_app/static/sample_input.csv` and is
downloadable from the **Download sample CSV** button on the UI.

---

## Architecture notes

- **Feature pipeline** (22 engineered indicators): 1-day & log returns, SMA(5/10/20)
  with close/SMA ratios, volatility windows, momentum (3/5/10), RSI(14), MACD +
  signal + histogram, Bollinger position, volume ratio, intraday range and close
  position.
- **Chronological splits** per ticker (70 / 15 / 15) prevent leakage of future
  data into training.
- **LSTM window** of 20 days yields `(20, 22)` sequence tensors; the sklearn
  models consume only the final day's feature vector.
- **Ensemble summary** in the UI averages the three direction probabilities and
  three magnitude predictions; per-model outputs are always shown side-by-side
  for transparency.

## Evaluation metrics

- **Classification** — Accuracy, Precision, Recall, F1
- **Regression**     — MAE, RMSE, R²

The training notebook's final leaderboards and the `metadata.json` artifact
carry these numbers for reference inside the deployed app.

---

## Disclaimer

NextTick is a coursework prototype. Forecasts are illustrative and must not be
treated as investment advice.
