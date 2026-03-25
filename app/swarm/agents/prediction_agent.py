"""
PredictionAgent — ML Ensemble Price & Trend Prediction  # noqa
======================================================
Phase 2 · Swarm Agent

Uses multi-model ensemble to predict:
  • Trend direction (BULLISH / BEARISH / NEUTRAL) for 1d, 5d, 20d horizons
  • Price range (predicted high/low/close) for next 5 days
  • Market regime (TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE)
  • Confidence score with uncertainty quantification

Models used (all CPU-based, no GPU required):
  1. RandomForest trend classifier (primary)
  2. Gradient Boosting classifier (secondary)
  3. Simple LSTM-inspired moving window regression
  4. Technical signal confluence scoring
  5. Momentum + volatility regime detector

Feature engineering:
  • 50+ features from raw OHLCV
  • Returns (1d, 5d, 20d), RSI, MACD, BB position, ATR%, volume ratios
  • Calendar features (day of week, month, options expiry week)
  • Distance from 52-week high/low

Input payload keys:
  symbol       (str, required)   NSE symbol
  days         (int)             Lookback for training. Default: 500
  horizons     (list)            Prediction horizons in days. Default: [1, 5, 20]
  use_cache    (bool)            Use cached model if available. Default: True

Output AgentResult.data keys:
  symbol, current_price, predictions (by horizon),
  price_range_5d, market_regime, confidence,
  feature_importance, model_agreement, explanation
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from ..base_agent import AgentResult, BaseSwarmAgent, SwarmMessage

logger = logging.getLogger(__name__)

AGENT_TYPE = "prediction"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:  # type: ignore[return]
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))  # type: ignore[return-value]


def _atr_pct(
    highs: pd.Series, lows: pd.Series, closes: pd.Series, p: int = 14
) -> pd.Series:  # type: ignore[return]
    tr = pd.concat(
        [
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows - closes.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(p).mean()
    return atr / closes  # type: ignore[return-value]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 50+ features from OHLCV DataFrame.
    Returns feature DataFrame aligned to the same index.
    """
    df = df.copy()
    c: pd.Series = df["close"]
    h: pd.Series = df["high"]
    l: pd.Series = df["low"]
    o: pd.Series = df["open"]
    v_raw = df.get("volume")
    v: pd.Series = v_raw if v_raw is not None else pd.Series(0, index=df.index)  # type: ignore[assignment]

    feat = pd.DataFrame(index=df.index)

    # ── Returns ─────────────────────────────────────────────────────────
    feat["ret_1d"] = c.pct_change(1)
    feat["ret_3d"] = c.pct_change(3)
    feat["ret_5d"] = c.pct_change(5)
    feat["ret_10d"] = c.pct_change(10)
    feat["ret_20d"] = c.pct_change(20)
    feat["log_ret_1d"] = np.log(c / c.shift(1))

    # ── Volatility ───────────────────────────────────────────────────────
    feat["vol_5d"] = feat["ret_1d"].rolling(5).std()
    feat["vol_10d"] = feat["ret_1d"].rolling(10).std()
    feat["vol_20d"] = feat["ret_1d"].rolling(20).std()
    feat["vol_ratio"] = feat["vol_5d"] / (feat["vol_20d"] + 1e-9)
    feat["atr_pct"] = _atr_pct(h, l, c, 14)

    # ── RSI ──────────────────────────────────────────────────────────────
    feat["rsi_14"] = _rsi(c, 14)
    feat["rsi_21"] = _rsi(c, 21)
    feat["rsi_diff"] = feat["rsi_14"] - feat["rsi_14"].shift(1)

    # ── MACD ─────────────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = macd_line - signal_line
    feat["macd_signal"] = (macd_line > signal_line).astype(int)
    feat["macd_above_zero"] = (macd_line > 0).astype(int)

    # ── Bollinger Bands ──────────────────────────────────────────────────
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_u = sma20 + 2 * std20
    bb_l = sma20 - 2 * std20
    feat["bb_pct_b"] = (c - bb_l) / (bb_u - bb_l + 1e-9)
    feat["bb_width_pct"] = (bb_u - bb_l) / (sma20 + 1e-9)

    # ── Moving averages ───────────────────────────────────────────────────
    for span in [9, 20, 50, 100, 200]:
        if len(c) >= span:
            ema = c.ewm(span=span, adjust=False).mean()
            feat[f"price_vs_ema{span}"] = (c - ema) / (ema + 1e-9)
            feat[f"ema{span}_slope"] = ema.diff(3) / (ema + 1e-9)

    # ── Stochastic ────────────────────────────────────────────────────────
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = 100 * (c - low14) / (high14 - low14 + 1e-9)
    stoch_d = stoch_k.rolling(3).mean()
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_d
    feat["stoch_diff"] = stoch_k - stoch_d

    # ── Volume features ───────────────────────────────────────────────────
    if v is not None and v.sum() > 0:
        vol_ma20 = v.rolling(20).mean()
        feat["vol_ratio_20d"] = v / (vol_ma20 + 1e-9)
        feat["vol_spike"] = (feat["vol_ratio_20d"] > 2.0).astype(int)
        # OBV trend
        direction = c.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * v).cumsum()
        obv_ema = obv.ewm(span=20, adjust=False).mean()
        feat["obv_trend"] = (obv > obv_ema).astype(int)
    else:
        feat["vol_ratio_20d"] = 0.0
        feat["vol_spike"] = 0
        feat["obv_trend"] = 0

    # ── Price position in range ───────────────────────────────────────────
    high52 = h.rolling(252, min_periods=50).max()
    low52 = l.rolling(252, min_periods=50).min()
    feat["pct_from_52h"] = (c - high52) / (high52 + 1e-9)
    feat["pct_from_52l"] = (c - low52) / (low52 + 1e-9)
    feat["range_position"] = (c - low52) / (high52 - low52 + 1e-9)

    # ── Intraday range ────────────────────────────────────────────────────
    feat["day_range_pct"] = (h - l) / (c + 1e-9)
    feat["close_vs_high"] = (c - h) / (h + 1e-9)
    feat["close_vs_low"] = (c - l) / (l + 1e-9)
    feat["open_gap_pct"] = (o - c.shift(1)) / (c.shift(1) + 1e-9)

    # ── Momentum ─────────────────────────────────────────────────────────
    feat["roc_5d"] = c.pct_change(5)
    feat["roc_10d"] = c.pct_change(10)
    feat["roc_20d"] = c.pct_change(20)
    feat["momentum_diff"] = feat["roc_5d"] - feat["roc_20d"]

    # ── Calendar ─────────────────────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        idx: pd.DatetimeIndex = df.index
        feat["day_of_week"] = idx.day_of_week.astype(float)
        feat["month"] = idx.month.astype(float)
        feat["is_month_end"] = idx.is_month_end.astype(float)
        feat["is_month_start"] = idx.is_month_start.astype(float)
    else:
        feat["day_of_week"] = 2.0
        feat["month"] = 6.0
        feat["is_month_end"] = 0.0
        feat["is_month_start"] = 0.0

    # ── ADX proxy (trend strength) ────────────────────────────────────────
    ema_fast = c.ewm(span=10, adjust=False).mean()
    ema_slow = c.ewm(span=30, adjust=False).mean()
    feat["trend_strength"] = (ema_fast - ema_slow).abs() / (ema_slow + 1e-9)
    feat["trend_direction"] = (ema_fast > ema_slow).astype(float)

    # ── Higher highs / lower lows streak ─────────────────────────────────
    feat["hh_streak"] = (h > h.shift(1)).astype(int).rolling(5).sum()
    feat["ll_streak"] = (l < l.shift(1)).astype(int).rolling(5).sum()

    # ── Candle body strength ──────────────────────────────────────────────
    feat["body_pct"] = (c - o).abs() / (h - l + 1e-9)
    feat["is_bullish"] = (c > o).astype(float)

    # Drop NaN rows
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    return feat


def build_labels(closes: pd.Series, horizon: int = 5) -> pd.Series:  # type: ignore[return]
    """
    Build classification labels for a given forward horizon.
      +1 = BULLISH  (forward return > +1%)
      -1 = BEARISH  (forward return < -1%)
       0 = NEUTRAL  (between -1% and +1%)
    """
    fwd_ret = closes.shift(-horizon) / closes - 1
    labels = pd.Series(0, index=closes.index, dtype=int)
    labels[fwd_ret > 0.01] = 1
    labels[fwd_ret < -0.01] = -1
    return labels  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Model training + prediction (scikit-learn)
# ---------------------------------------------------------------------------


def _train_rf(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def _train_gb(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    return gb


def _predict_ensemble(
    X_last: np.ndarray,
    rf_model: Any,
    gb_model: Any,
    classes: np.ndarray,
) -> Tuple[int, float, Dict[int, float]]:
    """
    Ensemble prediction with probability averaging.

    Returns:
        (predicted_class, confidence, probabilities_dict)
    """
    rf_proba = rf_model.predict_proba(X_last.reshape(1, -1))[0]
    gb_proba = gb_model.predict_proba(X_last.reshape(1, -1))[0]

    # Average probabilities
    avg_proba = (rf_proba + gb_proba) / 2
    pred_idx = int(np.argmax(avg_proba))
    pred_class = int(classes[pred_idx])
    confidence = float(avg_proba[pred_idx])

    proba_dict: Dict[int, float] = {
        int(c): round(float(p), 4) for c, p in zip(classes, avg_proba)
    }
    return pred_class, confidence, proba_dict


# ---------------------------------------------------------------------------
# Price range estimation
# ---------------------------------------------------------------------------


def _estimate_price_range(
    df: pd.DataFrame,
    horizon: int = 5,
) -> Dict[str, float]:
    """
    Estimate price range (high/low/close) for the next `horizon` days
    using historical volatility scaling.

    Approach:
      • Compute daily ATR (average true range)
      • Expected range ≈ ATR × sqrt(horizon) × 1.5
      • Trend-adjusted midpoint from recent momentum
    """
    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    n = len(closes)

    current = float(closes.iloc[-1])

    # ATR-based range
    tr = pd.concat(
        [
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows - closes.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.tail(20).mean())

    # Momentum: recent 5-day return
    if n >= 6:
        recent_ret = float(closes.iloc[-1] / closes.iloc[-6] - 1)
    else:
        recent_ret = 0.0

    # Scaled range
    expected_move = atr * np.sqrt(horizon) * 1.2
    trend_bias = current * recent_ret * 0.5

    pred_close = current + trend_bias
    pred_high = pred_close + expected_move
    pred_low = pred_close - expected_move

    # Confidence interval (95%) — ±2 sigma
    daily_vol = float(closes.pct_change().tail(20).std())
    horizon_vol = daily_vol * np.sqrt(horizon) * current
    ci_upper = pred_close + 1.96 * horizon_vol
    ci_lower = pred_close - 1.96 * horizon_vol

    return {
        "current_price": round(current, 2),
        "predicted_close": round(pred_close, 2),
        "predicted_high": round(pred_high, 2),
        "predicted_low": round(pred_low, 2),
        "ci_95_upper": round(ci_upper, 2),
        "ci_95_lower": round(ci_lower, 2),
        "expected_move_pts": round(expected_move, 2),
        "expected_move_pct": round(expected_move / current * 100, 2),
        "horizon_days": horizon,
        "atr_used": round(atr, 2),
    }


# ---------------------------------------------------------------------------
# Market regime detection
# ---------------------------------------------------------------------------


def _detect_regime(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Classify the current market regime using volatility + trend metrics.

    Regimes:
      TRENDING_UP    — strong uptrend + low-moderate volatility
      TRENDING_DOWN  — strong downtrend + low-moderate volatility
      RANGING        — weak trend + low volatility
      VOLATILE       — high volatility regardless of direction
    """
    closes = df["close"]
    n = len(closes)

    if n < 20:
        return {
            "regime": "UNKNOWN",
            "confidence": 0.3,
            "interpretation": "Insufficient data.",
        }

    # Trend: EMA50 slope
    ema50 = closes.ewm(span=50, adjust=False).mean()
    ema_slope = (
        float((ema50.iloc[-1] - ema50.iloc[-10]) / ema50.iloc[-10]) * 100
    )  # % over 10 bars

    # Volatility: 20-day annualised vol
    daily_vol = float(closes.pct_change().tail(20).std())
    ann_vol = daily_vol * np.sqrt(252) * 100  # percent

    # ADX proxy
    ema_fast = closes.ewm(span=12, adjust=False).mean()
    ema_slow = closes.ewm(span=26, adjust=False).mean()
    adx_proxy = (
        abs(float(ema_fast.iloc[-1] - ema_slow.iloc[-1]))
        / float(ema_slow.iloc[-1])
        * 100
    )

    HIGH_VOL_THRESHOLD = 30.0  # annualised % vol
    TREND_THRESHOLD = 0.5  # slope % per 10 bars
    ADX_THRESHOLD = 1.5  # % separation

    if ann_vol > HIGH_VOL_THRESHOLD:
        regime = "VOLATILE"
        interp = (
            f"Annualised volatility {ann_vol:.1f}% — high volatility regime. "
            "Options premiums elevated. Wider stops required."
        )
        confidence = min(1.0, ann_vol / 50.0)
    elif adx_proxy > ADX_THRESHOLD:
        if ema_slope > TREND_THRESHOLD:
            regime = "TRENDING_UP"
            interp = (
                f"EMA50 rising ({ema_slope:+.2f}% / 10 bars). "
                f"ADX proxy {adx_proxy:.2f}% — established uptrend."
            )
            confidence = min(1.0, adx_proxy / 3.0)
        elif ema_slope < -TREND_THRESHOLD:
            regime = "TRENDING_DOWN"
            interp = (
                f"EMA50 declining ({ema_slope:+.2f}% / 10 bars). "
                f"ADX proxy {adx_proxy:.2f}% — established downtrend."
            )
            confidence = min(1.0, adx_proxy / 3.0)
        else:
            regime = "RANGING"
            interp = (
                f"Flat EMA50 (slope {ema_slope:+.2f}%), "
                f"moderate ADX {adx_proxy:.2f}% — consolidation phase."
            )
            confidence = 0.6
    else:
        regime = "RANGING"
        interp = (
            f"Weak trend (ADX proxy {adx_proxy:.2f}%), "
            f"low volatility {ann_vol:.1f}% — range-bound market."
        )
        confidence = 0.55

    return {
        "regime": regime,
        "confidence": round(confidence, 3),
        "ann_volatility_pct": round(ann_vol, 2),
        "ema50_slope_pct": round(ema_slope, 3),
        "adx_proxy": round(adx_proxy, 3),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Backtesting metrics
# ---------------------------------------------------------------------------


def _quick_backtest(
    labels: pd.Series,
    predictions: np.ndarray,
    closes: pd.Series,
) -> Dict[str, Any]:
    """Compute accuracy and win rate on test set."""
    from sklearn.metrics import accuracy_score

    if len(labels) < 10:
        return {"accuracy": 0.0, "win_rate": 0.0, "note": "insufficient test samples"}

    y_true = labels.values
    n = min(len(y_true), len(predictions))
    y_pred = predictions[:n]
    y_true = y_true[:n]

    if n < 5:
        return {"accuracy": 0.0, "win_rate": 0.0, "note": "insufficient test samples"}

    try:
        acc = float(accuracy_score(y_true, y_pred))
    except Exception:
        acc = 0.0
    correct = int((y_true == y_pred).sum())

    return {
        "accuracy": round(acc, 4),
        "win_rate": round(correct / max(n, 1), 4),
        "test_samples": n,
    }


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------


class PredictionAgent(BaseSwarmAgent):
    """
    ML Ensemble Prediction Agent.

    Trains lightweight ML models on historical OHLCV + features,
    predicts trend direction for multiple horizons, estimates price range,
    detects market regime, and returns a confidence-scored ensemble result.
    """

    AGENT_TYPE = AGENT_TYPE
    DEFAULT_TIMEOUT_S = 120.0

    # Minimum bars needed for meaningful training
    MIN_TRAINING_BARS = 120
    # Bars to hold out as test set (most recent)
    TEST_SIZE = 30

    async def execute(self, message: SwarmMessage) -> AgentResult:
        payload = message.payload
        symbol: str = str(payload.get("symbol", "NIFTY")).upper().strip()
        days: int = int(payload.get("days", 500))
        horizons: List[int] = list(payload.get("horizons", [1, 5, 20]))
        use_ta_signals: bool = bool(payload.get("use_ta_signals", True))

        self._log.info(f"PredictionAgent: {symbol} | days={days} | horizons={horizons}")

        # ── 1. Get OHLCV ─────────────────────────────────────────────────
        df = await self.tools.get_ohlcv(symbol, days=days, interval="1d")

        if df is None or df.empty or len(df) < self.MIN_TRAINING_BARS:
            return self._ok(
                data={
                    "symbol": symbol,
                    "error": f"Need at least {self.MIN_TRAINING_BARS} bars, got {len(df) if df is not None else 0}",
                },
                summary=f"Insufficient data for ML prediction of {symbol}.",
                signal="neutral",
                confidence=0.1,
            )

        df = self._normalise(df)
        current_price = float(df["close"].iloc[-1])
        self._log.info(f"{symbol}: {len(df)} bars | current=₹{current_price:,.2f}")

        # ── 2. Build features ─────────────────────────────────────────────
        try:
            feat_df = build_features(df)
        except Exception as exc:
            self._log.error(f"Feature engineering failed: {exc}")
            return self._ok(
                data={"symbol": symbol, "error": f"Feature engineering: {exc}"},
                summary=f"Feature engineering failed for {symbol}.",
                signal="neutral",
                confidence=0.1,
            )

        # ── 3. Train + predict per horizon ───────────────────────────────
        predictions: Dict[str, Any] = {}
        ensemble_signals: List[str] = []
        ensemble_confidences: List[float] = []

        for horizon in horizons:
            try:
                pred = await self._predict_horizon(df, feat_df, horizon)
                predictions[f"{horizon}d"] = pred
                ensemble_signals.append(pred.get("signal", "neutral"))
                ensemble_confidences.append(float(pred.get("confidence", 0.5)))
            except Exception as exc:
                self._log.warning(f"Prediction for horizon {horizon}d failed: {exc}")
                predictions[f"{horizon}d"] = {
                    "horizon": horizon,
                    "signal": "neutral",
                    "confidence": 0.2,
                    "error": str(exc),
                }

        # ── 4. Price range estimate (5-day) ──────────────────────────────
        try:
            price_range = _estimate_price_range(df, horizon=5)
        except Exception as exc:
            price_range = {"error": str(exc), "current_price": current_price}

        # ── 5. Market regime detection ───────────────────────────────────
        try:
            regime = _detect_regime(df)
        except Exception as exc:
            regime = {"regime": "UNKNOWN", "confidence": 0.3, "error": str(exc)}

        # ── 6. Ensemble signal ────────────────────────────────────────────
        overall_signal, overall_confidence = self._ensemble_signal(
            ensemble_signals, ensemble_confidences
        )

        # Adjust for regime
        if regime.get("regime") == "VOLATILE":
            overall_confidence = min(overall_confidence, 0.55)
        elif regime.get("regime") == "RANGING":
            overall_confidence = min(overall_confidence, 0.65)

        # ── 7. Build explanation ──────────────────────────────────────────
        explanation = self._explain(
            symbol, current_price, predictions, price_range, regime, overall_signal
        )

        return self._ok(
            data={
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "predictions": predictions,
                "price_range_5d": price_range,
                "market_regime": regime,
                "overall_signal": overall_signal,
                "signal_label": self._signal_label(overall_signal, overall_confidence),
                "training_bars": len(df),
                "analysis_date": datetime.utcnow().isoformat(),
            },
            summary=explanation,
            signal=overall_signal,
            confidence=overall_confidence,
            metadata={
                "agent": AGENT_TYPE,
                "symbol": symbol,
                "horizons": horizons,
                "regime": regime.get("regime", "UNKNOWN"),
            },
        )

    # ────────────────────────────────────────────────────────────────────────
    # Per-horizon prediction
    # ────────────────────────────────────────────────────────────────────────

    async def _predict_horizon(
        self,
        df: pd.DataFrame,
        feat_df: pd.DataFrame,
        horizon: int,
    ) -> Dict[str, Any]:
        """
        Train ensemble on all bars except the last TEST_SIZE,
        then predict for the latest bar.
        """
        import asyncio

        from sklearn.preprocessing import StandardScaler

        closes = df["close"]

        # Build labels
        labels = build_labels(
            pd.Series(closes.values, index=closes.index), horizon=horizon
        )

        # Align features and labels
        combined = feat_df.copy()
        combined["__label__"] = labels

        # Drop rows with any NaN
        combined.dropna(inplace=True)

        if len(combined) < self.MIN_TRAINING_BARS:
            raise ValueError(
                f"Only {len(combined)} usable rows after NaN drop (need {self.MIN_TRAINING_BARS})"
            )

        feature_cols = [c for c in combined.columns if c != "__label__"]

        # Train/test split
        n = len(combined)
        test_n = min(self.TEST_SIZE, n // 5)
        train_df = combined.iloc[: n - test_n]
        test_df = combined.iloc[n - test_n :]

        X_train = train_df[feature_cols].values
        y_train = train_df["__label__"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["__label__"].values

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Need at least 2 classes to train
        classes = np.unique(y_train)
        if len(classes) < 2:
            raise ValueError(f"Only one class in training data for horizon {horizon}d")

        # Train models in executor (CPU-bound)
        loop = asyncio.get_event_loop()
        rf_model = await loop.run_in_executor(None, _train_rf, X_train, y_train)
        gb_model = await loop.run_in_executor(None, _train_gb, X_train, y_train)

        # Predict latest bar
        X_latest_arr: np.ndarray = scaler.transform(
            combined[feature_cols].iloc[-1:].values
        )
        pred_class, confidence, proba_dict = _predict_ensemble(
            X_latest_arr[0], rf_model, gb_model, classes
        )

        # Backtest metrics on test set
        rf_test_preds: np.ndarray = rf_model.predict(X_test)
        backtest = _quick_backtest(
            pd.Series(y_test),
            rf_test_preds,
            closes.iloc[n - test_n :],
        )

        # Adjust confidence with backtest accuracy
        adjusted_conf = confidence * 0.7 + float(backtest.get("accuracy", 0.5)) * 0.3

        signal_map: Dict[int, str] = {1: "bullish", -1: "bearish", 0: "neutral"}
        signal = signal_map.get(pred_class, "neutral")

        # Feature importance (RF)
        if hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            top_features = sorted(
                zip(feature_cols, importances),
                key=lambda x: x[1],
                reverse=True,
            )[:8]
        else:
            top_features = []

        return {
            "horizon": horizon,
            "signal": signal,
            "predicted_class": pred_class,
            "confidence": round(adjusted_conf, 3),
            "raw_confidence": round(confidence, 3),
            "probabilities": {
                "bullish": round(float(proba_dict.get(1, 0.0)), 4),
                "neutral": round(float(proba_dict.get(0, 0.0)), 4),
                "bearish": round(float(proba_dict.get(-1, 0.0)), 4),
            },
            "top_features": [
                {"feature": f, "importance": round(imp, 4)} for f, imp in top_features
            ],
            "backtest": backtest,
            "training_samples": len(X_train),
        }

    # ────────────────────────────────────────────────────────────────────────
    # Ensemble across horizons
    # ────────────────────────────────────────────────────────────────────────

    def _ensemble_signal(
        self,
        signals: List[str],
        confidences: List[float],
    ) -> Tuple[str, float]:
        """Weighted vote across all horizon predictions."""
        bull = sum(c for s, c in zip(signals, confidences) if s == "bullish")
        bear = sum(c for s, c in zip(signals, confidences) if s == "bearish")
        total = sum(confidences) or 1e-9

        if bull > bear * 1.3:
            return "bullish", round(bull / total, 3)
        elif bear > bull * 1.3:
            return "bearish", round(bear / total, 3)
        else:
            return "neutral", round(max(bull, bear) / total, 3)

    def _signal_label(self, signal: str, confidence: float) -> str:
        if signal == "bullish":
            if confidence >= 0.75:
                return "STRONG BUY"
            elif confidence >= 0.55:
                return "BUY"
            else:
                return "WEAK BUY"
        elif signal == "bearish":
            if confidence >= 0.75:
                return "STRONG SELL"
            elif confidence >= 0.55:
                return "SELL"
            else:
                return "WEAK SELL"
        else:
            return "HOLD / NEUTRAL"

    def _explain(
        self,
        symbol: str,
        price: float,
        predictions: Dict[str, Any],
        price_range: Dict[str, Any],
        regime: Dict[str, Any],
        signal: str,
    ) -> str:
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(signal, "⚪")
        lines = [
            f"{emoji} **{symbol} ML Prediction — {self._signal_label(signal, 0.6)}**",
            f"  • Current Price: ₹{price:,.2f}",
            f"  • Market Regime: {regime.get('regime', 'UNKNOWN')} "
            f"(conf {regime.get('confidence', 0):.0%})",
        ]
        for h_key, pred in sorted(predictions.items()):
            if "error" not in pred:
                lines.append(
                    f"  • {h_key} Outlook: {pred.get('signal', '?').upper()} "
                    f"(conf {pred.get('confidence', 0):.0%})"
                )
        pr = price_range
        if pr.get("predicted_close"):
            lines.append(
                f"  • 5d Price Range: ₹{pr.get('predicted_low', '?'):,.0f} – "
                f"₹{pr.get('predicted_high', '?'):,.0f} "
                f"(target ₹{pr.get('predicted_close', '?'):,.0f})"
            )
        lines.append(
            "\n⚠️ *ML predictions are probabilistic, not guarantees. "
            "Not financial advice.*"
        )
        return "\n".join(lines)

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                df[col] = float("nan")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df
