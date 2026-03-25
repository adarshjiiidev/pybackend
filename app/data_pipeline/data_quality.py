"""
DataQualityChecker — Validates, cleans, and repairs OHLCV time-series data.

Checks performed:
  1. Schema validation  — required columns present and numeric
  2. OHLC integrity     — H >= max(O,C), L <= min(O,C)
  3. Outlier detection  — Z-score based price spike detection
  4. Gap detection      — missing trading days in the series
  5. Gap filling        — forward-fill / interpolation for minor gaps
  6. Duplicate removal  — deduplicate on timestamp index
  7. Stale data check   — warn if latest bar is older than expected

Usage::

    checker = DataQualityChecker()
    report = checker.validate(df, symbol="RELIANCE")
    clean_df = checker.clean(df, symbol="RELIANCE")
    full_report, clean_df = checker.validate_and_clean(df, symbol="RELIANCE")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: List[str] = ["open", "high", "low", "close"]
OPTIONAL_COLUMNS: List[str] = ["volume"]

# Z-score threshold beyond which a bar is flagged as a price spike
ZSCORE_OUTLIER_THRESHOLD = 4.5

# Maximum allowed gap in calendar days before flagging as a data gap
# (weekends + 2 NSE holidays buffer)
MAX_ALLOWED_GAP_DAYS = 5

# Minimum bars required for meaningful analysis
MIN_BARS_FOR_ANALYSIS = 20

# Maximum forward-fill gap (bars) before we consider data missing rather than gap-fill
MAX_FFILL_BARS = 3


# ---------------------------------------------------------------------------
# Validation result helpers
# ---------------------------------------------------------------------------


class QualityReport:
    """Structured quality report for a single symbol's OHLCV DataFrame."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.fixes_applied: List[str] = []
        self.original_rows: int = 0
        self.final_rows: int = 0
        self.outliers_detected: int = 0
        self.gaps_detected: int = 0
        self.duplicates_removed: int = 0
        self.ohlc_violations_fixed: int = 0
        self.is_valid: bool = True

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_fix(self, msg: str) -> None:
        self.fixes_applied.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "is_valid": self.is_valid,
            "original_rows": self.original_rows,
            "final_rows": self.final_rows,
            "errors": self.errors,
            "warnings": self.warnings,
            "fixes_applied": self.fixes_applied,
            "stats": {
                "outliers_detected": self.outliers_detected,
                "gaps_detected": self.gaps_detected,
                "duplicates_removed": self.duplicates_removed,
                "ohlc_violations_fixed": self.ohlc_violations_fixed,
            },
        }

    def summary(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return (
            f"[{self.symbol}] {status} | rows {self.original_rows}→{self.final_rows} | "
            f"errors={len(self.errors)} warns={len(self.warnings)} fixes={len(self.fixes_applied)}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DataQualityChecker:
    """
    Validates and cleans OHLCV DataFrames before storing in the data pool
    or feeding into analytics / prediction engines.

    All methods are synchronous (CPU-bound pandas work — no I/O).
    """

    def __init__(
        self,
        zscore_threshold: float = ZSCORE_OUTLIER_THRESHOLD,
        max_gap_days: int = MAX_ALLOWED_GAP_DAYS,
        min_bars: int = MIN_BARS_FOR_ANALYSIS,
    ):
        self.zscore_threshold = zscore_threshold
        self.max_gap_days = max_gap_days
        self.min_bars = min_bars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> QualityReport:
        """
        Run all validation checks on a DataFrame and return a QualityReport.
        Does NOT modify the DataFrame.

        Args:
            df:     OHLCV DataFrame with DatetimeIndex
            symbol: Symbol name (for logging / report labelling)

        Returns:
            QualityReport with errors, warnings, and stats
        """
        report = QualityReport(symbol)
        report.original_rows = len(df)

        if df.empty:
            report.add_error("DataFrame is empty")
            return report

        self._check_schema(df, report)
        if not report.is_valid:
            return report  # can't proceed without required columns

        self._check_min_bars(df, report)
        self._check_index(df, report)
        self._check_duplicates(df, report)
        self._check_ohlc_integrity(df, report)
        self._check_outliers(df, report)
        self._check_gaps(df, report)
        self._check_stale_data(df, report)
        self._check_zero_prices(df, report)
        self._check_negative_volume(df, report)

        report.final_rows = len(df)
        logger.debug(report.summary())
        return report

    def clean(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        fill_gaps: bool = True,
        remove_outliers: bool = True,
        fix_ohlc: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all cleaning operations to the DataFrame and return a clean copy.

        Operations (in order):
          1. Normalise column names to lowercase
          2. Parse and sort DatetimeIndex
          3. Remove duplicate timestamps (keep last)
          4. Fix OHLC integrity violations
          5. Remove or cap outlier bars
          6. Fill small gaps (forward-fill up to MAX_FFILL_BARS)
          7. Drop rows with any NaN in OHLC columns

        Args:
            df:              Source DataFrame
            symbol:          For logging
            fill_gaps:       Whether to forward-fill minor gaps
            remove_outliers: Whether to remove detected price spikes
            fix_ohlc:        Whether to auto-fix minor OHLC violations

        Returns:
            Cleaned DataFrame (copy — original is unchanged)
        """
        if df.empty:
            logger.warning(f"clean(): empty DataFrame for {symbol}")
            return df.copy()

        df = df.copy()
        df = self._normalise_columns(df)

        if df.empty or not all(c in df.columns for c in REQUIRED_COLUMNS):
            logger.error(f"clean(): missing required columns for {symbol}")
            return df

        df = self._ensure_datetime_index(df)
        df = self._sort_index(df)
        df = self._remove_duplicates(df, symbol)

        if fix_ohlc:
            df = self._fix_ohlc_violations(df, symbol)

        if remove_outliers:
            df = self._remove_outlier_bars(df, symbol)

        if fill_gaps:
            df = self._fill_small_gaps(df, symbol)

        # Final NaN drop
        before = len(df)
        df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
        dropped = before - len(df)
        if dropped > 0:
            logger.debug(f"clean({symbol}): dropped {dropped} rows with NaN OHLC")

        # Ensure volume column exists
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["volume"] = df["volume"].fillna(0.0)

        return df

    def validate_and_clean(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> Tuple[QualityReport, pd.DataFrame]:
        """
        Convenience method: validate → clean → validate again.

        Returns:
            (report_after_cleaning, clean_df)
        """
        clean_df = self.clean(df, symbol=symbol)
        report = self.validate(clean_df, symbol=symbol)
        report.original_rows = len(df)
        report.final_rows = len(clean_df)
        return report, clean_df

    # ------------------------------------------------------------------
    # Validation checks (read-only)
    # ------------------------------------------------------------------

    def _check_schema(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Verify required columns exist and are numeric."""
        cols = [str(c).lower() for c in df.columns]
        for col in REQUIRED_COLUMNS:
            if col not in cols:
                report.add_error(f"Missing required column: '{col}'")
                return

        for col in REQUIRED_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                report.add_error(
                    f"Column '{col}' is not numeric (dtype={df[col].dtype})"
                )

    def _check_min_bars(self, df: pd.DataFrame, report: QualityReport) -> None:
        if len(df) < self.min_bars:
            report.add_warning(
                f"Only {len(df)} bars available — minimum {self.min_bars} recommended for analysis"
            )

    def _check_index(self, df: pd.DataFrame, report: QualityReport) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            report.add_error(
                f"Index is {type(df.index).__name__}, expected DatetimeIndex"
            )
        elif df.index.hasnans:
            report.add_error("DatetimeIndex contains NaT values")

    def _check_duplicates(self, df: pd.DataFrame, report: QualityReport) -> None:
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            report.add_warning(f"{dupes} duplicate timestamps found in index")
            report.duplicates_removed = dupes

    def _check_ohlc_integrity(self, df: pd.DataFrame, report: QualityReport) -> None:
        """
        Validate:
          • High  ≥ Open  AND  High  ≥ Close
          • Low   ≤ Open  AND  Low   ≤ Close
          • High  ≥ Low
          • All prices > 0
        """
        violations = 0

        # High must be >= all other prices
        h_vs_o = (df["high"] < df["open"]).sum()
        h_vs_c = (df["high"] < df["close"]).sum()
        h_vs_l = (df["high"] < df["low"]).sum()
        l_vs_o = (df["low"] > df["open"]).sum()
        l_vs_c = (df["low"] > df["close"]).sum()
        non_pos = (df["close"] <= 0).sum()

        violations = h_vs_o + h_vs_c + h_vs_l + l_vs_o + l_vs_c

        if h_vs_o > 0:
            report.add_warning(f"OHLC: {h_vs_o} bars where High < Open")
        if h_vs_c > 0:
            report.add_warning(f"OHLC: {h_vs_c} bars where High < Close")
        if h_vs_l > 0:
            report.add_warning(f"OHLC: {h_vs_l} bars where High < Low (critical)")
        if l_vs_o > 0:
            report.add_warning(f"OHLC: {l_vs_o} bars where Low > Open")
        if l_vs_c > 0:
            report.add_warning(f"OHLC: {l_vs_c} bars where Low > Close")
        if non_pos > 0:
            report.add_error(f"OHLC: {non_pos} bars with Close price ≤ 0")

        report.ohlc_violations_fixed = violations

    def _check_outliers(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Detect price spikes using Z-score on log returns."""
        if len(df) < 10:
            return
        try:
            log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
            if log_returns.std() == 0:
                return
            z_scores = (log_returns - log_returns.mean()) / log_returns.std()
            outliers = (np.abs(z_scores) > self.zscore_threshold).sum()
            if outliers > 0:
                report.outliers_detected = int(outliers)
                report.add_warning(
                    f"{outliers} potential outlier bars detected "
                    f"(Z-score > {self.zscore_threshold:.1f})"
                )
        except Exception as exc:
            logger.debug(f"Outlier check error: {exc}")

    def _check_gaps(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Detect abnormally large gaps in the time series."""
        if len(df) < 2:
            return
        diffs = df.index.to_series().diff().dropna()
        large_gaps = diffs[diffs > pd.Timedelta(days=self.max_gap_days)]
        if not large_gaps.empty:
            report.gaps_detected = len(large_gaps)
            max_gap = large_gaps.max()
            report.add_warning(
                f"{len(large_gaps)} data gaps detected (largest: {max_gap.days} calendar days)"
            )

    def _check_stale_data(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Warn if the latest bar is suspiciously old."""
        if df.empty:
            return
        latest = df.index.max()
        now = pd.Timestamp.utcnow().tz_localize(None)
        age_days = (
            (now - latest).days if hasattr(latest, "days") else (now - latest).days
        )
        # Allow up to 5 days (weekend + holiday buffer)
        if age_days > 5:
            report.add_warning(
                f"Latest bar is {age_days} days old ({latest.date()}) — "
                "data may be stale"
            )

    def _check_zero_prices(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Flag bars where any price is exactly zero (common data errors)."""
        for col in REQUIRED_COLUMNS:
            if col in df.columns:
                zeros = (df[col] == 0).sum()
                if zeros > 0:
                    report.add_warning(f"Column '{col}' has {zeros} zero values")

    def _check_negative_volume(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Flag negative volume values."""
        if "volume" in df.columns:
            neg = (df["volume"] < 0).sum()
            if neg > 0:
                report.add_warning(f"Volume has {neg} negative values")

    # ------------------------------------------------------------------
    # Cleaning operations (return modified copies)
    # ------------------------------------------------------------------

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase column names; handle MultiIndex flattening."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(c).lower() for c in col).strip("_") for col in df.columns
            ]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        return df

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce index to DatetimeIndex if needed."""
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:
                logger.warning(f"Could not convert index to DatetimeIndex: {exc}")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df

    def _sort_index(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_index()

    def _remove_duplicates(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the last entry."""
        n_before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        removed = n_before - len(df)
        if removed > 0:
            logger.debug(f"clean({symbol}): removed {removed} duplicate timestamp rows")
        return df

    def _fix_ohlc_violations(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Fix minor OHLC integrity violations:
          • If High < Low  → swap them
          • If High < Close → set High = Close
          • If High < Open  → set High = Open
          • If Low  > Close → set Low  = Close
          • If Low  > Open  → set Low  = Open
        """
        fixed = 0

        # Swap High < Low
        mask_hl = df["high"] < df["low"]
        if mask_hl.any():
            df.loc[mask_hl, ["high", "low"]] = df.loc[mask_hl, ["low", "high"]].values
            fixed += mask_hl.sum()

        # High must be >= Open and Close
        df["high"] = df[["high", "open", "close"]].max(axis=1)

        # Low must be <= Open and Close
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        if fixed > 0:
            logger.debug(f"clean({symbol}): fixed {fixed} H/L swap violations")

        return df

    def _remove_outlier_bars(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Remove bars where the close-to-close log return exceeds the Z-score threshold.
        These are almost certainly data errors (e.g. 10x price spikes).
        """
        if len(df) < 10:
            return df

        try:
            log_returns = np.log(df["close"] / df["close"].shift(1))
            mean = log_returns.mean()
            std = log_returns.std()

            if std == 0:
                return df

            z_scores = (log_returns - mean).abs() / std
            # Keep bar if Z-score < threshold OR it's the first bar (NaN)
            mask_keep = (z_scores < self.zscore_threshold) | z_scores.isna()
            removed = (~mask_keep).sum()

            if removed > 0:
                logger.debug(
                    f"clean({symbol}): removed {removed} outlier bars "
                    f"(Z > {self.zscore_threshold:.1f})"
                )
                df = df[mask_keep]
        except Exception as exc:
            logger.debug(f"Outlier removal error for {symbol}: {exc}")

        return df

    def _fill_small_gaps(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Forward-fill gaps of up to MAX_FFILL_BARS consecutive missing bars.

        We do NOT resample the entire series (that would create phantom weekend bars).
        Instead we identify internal NaN sequences created by prior cleaning and fill.
        """
        # After removing outliers, some internal NaN rows may exist; forward-fill them
        nan_count = df["close"].isna().sum()
        if nan_count > 0 and nan_count <= MAX_FFILL_BARS:
            df = df.ffill(limit=MAX_FFILL_BARS)
            logger.debug(f"clean({symbol}): forward-filled {nan_count} NaN bars")
        return df

    # ------------------------------------------------------------------
    # Utility / static methods
    # ------------------------------------------------------------------

    @staticmethod
    def compute_data_completeness(
        df: pd.DataFrame, expected_trading_days: int
    ) -> float:
        """
        Return fraction of expected trading days present in df.

        Args:
            df:                     OHLCV DataFrame with DatetimeIndex
            expected_trading_days:  How many trading days you'd expect in the window

        Returns:
            completeness ratio in [0.0, 1.0]
        """
        if expected_trading_days <= 0:
            return 1.0
        actual = len(df.dropna(subset=["close"]))
        return min(1.0, actual / expected_trading_days)

    @staticmethod
    def detect_split_adjusted_jumps(
        df: pd.DataFrame, threshold: float = 0.40
    ) -> List[pd.Timestamp]:
        """
        Detect timestamps where the close price jumped/dropped by > threshold (40%)
        in a single bar — often indicating an unadjusted stock split.

        Returns:
            List of timestamps where a potential split adjustment is needed
        """
        if "close" not in df.columns or len(df) < 2:
            return []
        pct_change = df["close"].pct_change().abs()
        suspicious = pct_change[pct_change > threshold]
        return list(suspicious.index)

    @staticmethod
    def compute_completeness_score(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return a comprehensive data quality score dict.

        Returns dict with:
          - completeness_pct: % of rows with no NaN in OHLC
          - ohlc_integrity_pct: % of bars with valid H >= L
          - volume_coverage_pct: % of bars with positive volume
          - total_bars: total row count
        """
        if df.empty:
            return {
                "completeness_pct": 0.0,
                "ohlc_integrity_pct": 0.0,
                "volume_coverage_pct": 0.0,
                "total_bars": 0,
            }

        total = len(df)
        cols_available = [c for c in REQUIRED_COLUMNS if c in df.columns]
        complete_rows = df[cols_available].notna().all(axis=1).sum()

        if "high" in df.columns and "low" in df.columns:
            valid_hl = (df["high"] >= df["low"]).sum()
        else:
            valid_hl = total

        if "volume" in df.columns:
            pos_volume = (df["volume"] > 0).sum()
        else:
            pos_volume = 0

        return {
            "completeness_pct": round(complete_rows / total * 100, 2),
            "ohlc_integrity_pct": round(valid_hl / total * 100, 2),
            "volume_coverage_pct": round(pos_volume / total * 100, 2),
            "total_bars": total,
        }
