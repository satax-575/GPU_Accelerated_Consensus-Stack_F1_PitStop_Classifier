"""
F1 Pit Stop Prediction - Preprocessing Pipeline

This file handles numeric conditioning of the engineered feature matrix.
It does three things: imputes any remaining NaNs (defensive), caps extreme
outliers via Winsorization, and applies the right scaler for each feature
group. It does NOT create new features or decide which features to keep.

Leakage note: all statistics (quantile bounds, scaler params) are learned
inside fit(). Always call fit() on training-fold rows only inside a CV loop.

Feature groups and why they're preprocessed differently:
  BINARY_FLAGS   - already 0/1, just pass through (or StandardScale for linear models)
  ORDINAL_INTS   - small integers, StandardScaler centers them for gradient models
  HIGH_SKEW_CONT - very skewed; Winsorize first, then Yeo-Johnson, then StandardScale
  BOUNDED_RATIO  - lives in [0,1] by design; MinMaxScaler keeps that meaning
  STD_CONT       - ordinary continuous; RobustScaler handles leftover outliers
  PASSTHROUGH    - label-encoded ints and compound dummies; tree models want them raw

"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore")

# Feature group definitions

BINARY_FLAGS: list[str] = [
    "PitStop",
    "past_cliff",
    "is_wet_compound",
    "is_soft_compound",
    "laptime_worsening",
    "prev_pitstop",
    "too_late_to_pit",
    "undercut_window",
    "overcut_candidate",
    "in_points",
    "cpd_HARD",
    "cpd_INTERMEDIATE",
    "cpd_MEDIUM",
    "cpd_SOFT",
    "cpd_WET",
    "race_phase_early",
    "race_phase_mid",
    "race_phase_late",
    "in_pit_window",
    "is_first_stint",
    "sc_lap_proxy",
    "is_cluster_lap",
    "sc_spike_flag",
]

ORDINAL_INTS: list[str] = [
    "Year",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "compound_softness",
    "compound_expected_stint",
    "compound_cliff_lap",
    "total_race_laps",
    "laps_remaining",
    "consecutive_worse",
    "pit_cluster_size",
    "recent_sc_proxy",
    "driver_enc",
    "race_enc",
]

HIGH_SKEW_CONT: list[str] = [
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "laptime_zscore",
    "laptime_delta_abs",
    "tyre_life_sq",
    "deg_per_lap",
    "deg_accel",
    "urgency_index",
    "cliff_proximity",
    "tyreage_x_progress",
    "momentum_x_overrun",
    "pace_momentum_3",
    "roll_lt_std_1",
    "roll_lt_std_3",
    "roll_lt_std_5",
    "laptime_lag1",
    "roll_lt_mean_1",
    "roll_lt_mean_3",
    "roll_lt_mean_5",
]

BOUNDED_RATIO: list[str] = [
    "RaceProgress",
    "laps_remaining_norm",
    "position_norm",
    "tyre_life_norm",
    "tyre_overrun_amount",
]

STD_CONT: list[str] = [
    "tyre_life_log",
    "laps_to_cliff",
    "laps_past_expected",
    "stint_exhaustion",
    "cpd_rank_x_life",
    "position_change_abs",
    "position_trend",
    "roll_pos_mean_3",
    "roll_delta_1",
    "roll_delta_3",
    "roll_delta_5",
    "roll_deg_1",
    "roll_deg_3",
    "roll_deg_5",
    "cliff_x_window",
    "Position_Change",
]

class _SafeImputer(BaseEstimator, TransformerMixin):
    """
    Median or mode imputation that returns a DataFrame with column names intact.
    This is a defensive layer - the feature matrix should already be clean by
    the time it gets here, but this catches anything that slips through.
    """

    def __init__(self, strategy: str = "median") -> None:
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y=None) -> "_SafeImputer":
        self._cols    = list(X.columns)
        self._imputer = SimpleImputer(strategy=self.strategy)
        self._imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self._imputer.transform(X)
        return pd.DataFrame(arr, columns=self._cols, index=X.index)

class _WinsorCapper(BaseEstimator, TransformerMixin):
    """
    Clips values to [q_low, q_high] quantile bounds learned from training data.
    Applied to HIGH_SKEW_CONT features before PowerTransformer so extreme
    outliers don't distort the transformation.
    """

    def __init__(self, q_low: float = 0.001, q_high: float = 0.999) -> None:
        self.q_low  = q_low
        self.q_high = q_high

    def fit(self, X: pd.DataFrame, y=None) -> "_WinsorCapper":
        self._cols  = list(X.columns)
        self._lower = X.quantile(self.q_low)
        self._upper = X.quantile(self.q_high)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self._cols:
            df[col] = df[col].clip(lower=self._lower[col], upper=self._upper[col])
        return df

class _DFScaler(BaseEstimator, TransformerMixin):
    """
    Thin wrapper around any sklearn scaler that keeps column names and index.
    Without this, sklearn scalers return numpy arrays and we lose column info.

    redundant and misleading. Removed it - sklearn's default BaseEstimator
    .fit_transform() calls fit() then transform() correctly.
    """
    def __init__(self, scaler) -> None:
        self.scaler = scaler

    def fit(self, X: pd.DataFrame, y=None) -> "_DFScaler":
        self._cols = list(X.columns)
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.scaler.transform(X)
        return pd.DataFrame(arr, columns=self._cols, index=X.index)

class F1Preprocessor(BaseEstimator, TransformerMixin):
    """
    Scales and conditions the F1 feature matrix before modeling.

    Each feature group gets the preprocessing treatment that makes sense
    for its distribution:
      BINARY_FLAGS   : impute mode, optionally StandardScale
      ORDINAL_INTS   : impute median, StandardScale
      HIGH_SKEW_CONT : impute median, Winsorize, Yeo-Johnson, StandardScale
      BOUNDED_RATIO  : impute median, MinMaxScale to [0, 1]
      STD_CONT       : impute median, RobustScale
      Extra columns  : passed through raw (tree models handle integers fine)

    Parameters
    ----------
    passthrough_binary : bool
        If True (default), binary columns come out as 0/1 integers unchanged.
    winsor_q : tuple[float, float]
        Quantile bounds for Winsorization on HIGH_SKEW_CONT features.
    """

    def __init__(
        self,
        passthrough_binary: bool = True,
        winsor_q: tuple = (0.001, 0.999),
    ) -> None:
        self.passthrough_binary = passthrough_binary
        self.winsor_q           = winsor_q
        self._is_fitted         = False

    def _binary_pipe(self) -> list:
        steps: list = [("impute", _SafeImputer(strategy="most_frequent"))]
        if not self.passthrough_binary:
            steps.append(("scale", _DFScaler(StandardScaler())))
        return steps

    def _ordinal_pipe(self) -> list:
        return [
            ("impute", _SafeImputer(strategy="median")),
            ("scale",  _DFScaler(StandardScaler())),
        ]

    def _high_skew_pipe(self) -> list:
        return [
            ("impute",  _SafeImputer(strategy="median")),
            ("winsor",  _WinsorCapper(self.winsor_q[0], self.winsor_q[1])),
            ("power",   _DFScaler(PowerTransformer(method="yeo-johnson", standardize=False))),
            ("scale",   _DFScaler(StandardScaler())),
        ]

    def _bounded_pipe(self) -> list:
        return [
            ("impute", _SafeImputer(strategy="median")),
            ("scale",  _DFScaler(MinMaxScaler(feature_range=(0, 1)))),
        ]

    def _std_pipe(self) -> list:
        return [
            ("impute", _SafeImputer(strategy="median")),
            ("scale",  _DFScaler(RobustScaler())),
        ]

    @staticmethod
    def _build_and_fit(steps: list, X: pd.DataFrame) -> Optional[Pipeline]:
        """Build and fit a Pipeline. Returns None if the column list is empty."""
        if not steps or X.empty or X.shape[1] == 0:
            return None
        pipe = Pipeline(steps)
        pipe.fit(X)
        return pipe

    def fit(self, X: pd.DataFrame, y=None) -> "F1Preprocessor":
        """
        Learn all scaling statistics from training-fold rows only.
        Must never be called on validation or test data.
        """
        self._binary_cols  = [c for c in BINARY_FLAGS   if c in X.columns]
        self._ordinal_cols = [c for c in ORDINAL_INTS   if c in X.columns]
        self._skew_cols    = [c for c in HIGH_SKEW_CONT if c in X.columns]
        self._bounded_cols = [c for c in BOUNDED_RATIO  if c in X.columns]
        self._std_cols     = [c for c in STD_CONT       if c in X.columns]

        # Any column not in a named group gets passed through as-is
        known = set(
            self._binary_cols + self._ordinal_cols +
            self._skew_cols   + self._bounded_cols + self._std_cols
        )
        self._extra_cols = [c for c in X.columns if c not in known]

        self._pipe_binary  = self._build_and_fit(self._binary_pipe(),    X[self._binary_cols])  if self._binary_cols  else None
        self._pipe_ordinal = self._build_and_fit(self._ordinal_pipe(),   X[self._ordinal_cols]) if self._ordinal_cols else None
        self._pipe_skew    = self._build_and_fit(self._high_skew_pipe(), X[self._skew_cols])    if self._skew_cols    else None
        self._pipe_bounded = self._build_and_fit(self._bounded_pipe(),   X[self._bounded_cols]) if self._bounded_cols else None
        self._pipe_std     = self._build_and_fit(self._std_pipe(),       X[self._std_cols])     if self._std_cols     else None

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned scaling to X.

        [203, 501, 899, ...]. All transformed parts are reset to 0..n-1 before
        pd.concat(), otherwise the concat silently fills with NaNs wherever
        indexes don't align.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        parts: list[pd.DataFrame] = []

        def _apply(pipe, cols):
            # instead of passing an empty frame into concat and causing shape issues.
            if pipe is not None and cols:
                transformed = pipe.transform(X[cols])
                parts.append(
                    pd.DataFrame(
                        transformed.values if hasattr(transformed, "values") else transformed,
                        columns=cols,
                    )
                )
        _apply(self._pipe_binary,  self._binary_cols)
        _apply(self._pipe_ordinal, self._ordinal_cols)
        _apply(self._pipe_skew,    self._skew_cols)
        _apply(self._pipe_bounded, self._bounded_cols)
        _apply(self._pipe_std,     self._std_cols)

        if self._extra_cols:
            parts.append(
                X[self._extra_cols].reset_index(drop=True)
            )
        if not parts:
            # No columns matched any group at all - return an empty frame
            return pd.DataFrame(index=range(len(X)))

        result = pd.concat(parts, axis=1)

        # Put columns back in the same order they came in
        original_order = [c for c in X.columns if c in result.columns]
        remaining      = [c for c in result.columns if c not in original_order]
        return result[original_order + remaining].reset_index(drop=True)

    def fit_transform(self, X: pd.DataFrame, y=None, **kw) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_groups(self) -> dict[str, list[str]]:
        """Show which columns ended up in which preprocessing group after fit()."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        return {
            "binary_flags":      self._binary_cols,
            "ordinal_ints":      self._ordinal_cols,
            "high_skew_cont":    self._skew_cols,
            "bounded_ratio":     self._bounded_cols,
            "std_cont":          self._std_cols,
            "extra_passthrough": self._extra_cols,
        }

    def check_output(self, X_transformed: pd.DataFrame) -> pd.DataFrame:
        """
        Quick sanity check on the preprocessed matrix.
        Returns null counts, min, max, mean, std per column.
        """
        summary          = X_transformed.describe().T
        summary["nulls"] = X_transformed.isnull().sum()
        return summary
