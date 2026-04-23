"""
NextTick — Flask inference app.

Accepts a stock ticker symbol, fetches the last 6 months of OHLCV data
from Yahoo Finance, runs six trained models, and returns next-day
direction + magnitude predictions.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict

from flask import Flask, jsonify, render_template, request

from utils.fetcher import fetch_ohlcv, fetch_ticker_info, search_tickers
from utils.inference import InferenceService

# -------------------------------------------------------------- #
# Configuration
# -------------------------------------------------------------- #
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexttick")


# -------------------------------------------------------------- #
# App factory
# -------------------------------------------------------------- #
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("NEXTTICK_SECRET", "dev-secret-change-me")

    service = InferenceService(MODELS_DIR)
    logger.info("Artifact status: %s", service.status)

    # ---------------------------------------------------------- #
    # Routes
    # ---------------------------------------------------------- #
    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            service_ready=service.is_ready,
            artifact_status=service.status,
            metadata=service.metadata,
        )

    @app.route("/search", methods=["GET"])
    def search():
        q = request.args.get("q", "").strip()
        if len(q) < 2:
            return jsonify([])
        return jsonify(search_tickers(q))

    @app.route("/predict", methods=["POST"])
    def predict():
        # Accept JSON body or form field
        if request.is_json:
            body   = request.get_json(force=True) or {}
            ticker = body.get("ticker", "").strip().upper()
        else:
            ticker = request.form.get("ticker", "").strip().upper()

        if not ticker:
            return jsonify(error="No ticker symbol provided."), 400

        # ----- fetch live data -------------------------------- #
        try:
            df, source = fetch_ohlcv(ticker)
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 400
        except Exception as exc:
            logger.exception("Data fetch failed for %s", ticker)
            return jsonify(error=f"Could not fetch data: {exc}"), 500

        # ----- company metadata ------------------------------- #
        info = fetch_ticker_info(ticker)

        # ----- inference -------------------------------------- #
        try:
            result = service.predict(df)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503
        except Exception as exc:
            logger.exception("Inference failed for %s", ticker)
            return jsonify(error=f"Inference failed: {exc}"), 500

        logger.info(
            "Prediction served | ticker=%s source=%s rows=%d direction=%s conf=%.3f mag=%+.3f%%",
            ticker, source, len(df),
            result.direction, result.direction_confidence or 0, result.magnitude_pct or 0,
        )

        payload = {
            "ticker":               ticker,
            "company_name":         info["name"],
            "exchange":             info["exchange"],
            "data_source":          source,
            "direction":            result.direction,
            "direction_confidence": result.direction_confidence,
            "magnitude_pct":        result.magnitude_pct,
            "last_close":           result.last_close,
            "next_close":           result.next_close,
            "classifications":      [asdict(p) for p in result.classifications],
            "regressions":          [asdict(p) for p in result.regressions],
            "history":              result.history,
            "data_summary":         result.data_summary,
            "features_snapshot":    result.features_snapshot,
            "ensemble_detail":      result.ensemble_detail,
        }
        return jsonify(payload)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            ready=service.is_ready,
            artifact_status=service.status,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
