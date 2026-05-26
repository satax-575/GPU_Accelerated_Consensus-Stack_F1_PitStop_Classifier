"""
F1 Pit Stop Prediction - Feature Engineering Transformer

This file's only job is to create new features from the raw data using
domain knowledge. It does NOT scale, normalize, impute, or select features -
those are handled by F1Preprocessor and ConsensusFeatureSelector respectively.

Leakage note: anything that touches the target column (PitNextLap) or aggregates
across drivers on the same lap is confined to fit(). fit() must always be called
on the training fold only inside a CV loop.

Features that are pure arithmetic on a single row or hardcoded constants are
computed in transform() and are safe to call anytime.

Typical usage:
    from f1_feature_transformer import F1FoldWiseWrapper

    wrapper = F1FoldWiseWrapper(n_splits=5, random_state=42)

    # Option A - manual CV loop:
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_feat, X_val_feat = wrapper.fit_transform_fold(
            X.iloc[tr_idx], y.iloc[tr_idx],
            X.iloc[val_idx], fold_index=fold,
        )

    # Option B - run the full loop in one call:
    train_feats, val_feats, tr_idxs, val_idxs = wrapper.run_cv(X, y)

    # For test inference, fit on the entire training set first:
    wrapper.fit_full(X_train, y_train)
    X_test_feat = wrapper.transform_test(X_test)

"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Tyre compound metadata: softness rank, how long a stint typically lasts,
# and what lap the degradation cliff usually hits. These are calibrated from
# historical pit distributions and then refined from training data in fit().
COMPOUND_META: dict[str, dict] = {
    "SOFT":         {"rank": 1, "expected_stint": 12, "cliff_lap": 16},
    "MEDIUM":       {"rank": 2, "expected_stint": 17, "cliff_lap": 22},
    "HARD":         {"rank": 3, "expected_stint": 21, "cliff_lap": 28},
    "INTERMEDIATE": {"rank": 4, "expected_stint": 18, "cliff_lap": 25},
    "WET":          {"rank": 5, "expected_stint": 13, "cliff_lap": 20},
}

# Columns that identify a unique driver stint within a race
_GROUP: list[str] = ["Race", "Year", "Driver"]

# All compound one-hot column names we expect after pd.get_dummies()
_COMPOUND_COLS: list[str] = [
    "cpd_HARD", "cpd_INTERMEDIATE", "cpd_MEDIUM", "cpd_SOFT", "cpd_WET",
]

# The target column - must never appear in the output feature matrix
_TARGET_COL = "PitNextLap"

class F1FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates a richer feature matrix from raw F1 lap data.

    This class only creates features. It does not scale, clip, or select columns.

    fit() needs to be called on training data to learn five things that would
    cause leakage if computed globally:
      1. Race-level mean lap time          (needed for laptime_zscore)
      2. Race total laps                   (needed for laps_remaining)
      3. Compound cliff-lap from pit rows  (target-derived, training only)
      4. Cross-driver pit cluster counts   (target-adjacent, training only)
      5. Driver and Race label encoders    (seen values only)

    Parameters
    ----------
    lag_windows : list[int]
        Rolling window sizes for lag features (default [1, 3, 5]).
    drop_raw_ids : bool
        Whether to drop raw string columns (Driver, Race, Compound) after encoding.
    verbose : bool
        Show tqdm progress bars.
    """

    def __init__(
        self,
        lag_windows: Optional[list[int]] = None,
        drop_raw_ids: bool = True,
        verbose: bool = True,
    ) -> None:
        self.lag_windows  = lag_windows or [1, 3, 5]
        self.drop_raw_ids = drop_raw_ids
        self.verbose      = verbose

        # These dicts are populated in fit() from training data
        self._race_total_laps:   dict[tuple, float] = {}
        self._race_mean_laptime: dict[tuple, float] = {}
        self._pit_cluster_map:   dict[tuple, float] = {}

        # Start from hardcoded priors; fit() will refine cliff_lap from actual pits
        self._compound_cliff: dict[str, int] = {
            k: v["cliff_lap"] for k, v in COMPOUND_META.items()
        }
        self._driver_enc = LabelEncoder()
        self._race_enc   = LabelEncoder()

        self._compound_expected: dict[str, int] = {
            k: v["expected_stint"] for k, v in COMPOUND_META.items()
        }
        self._compound_softness: dict[str, int] = {
            k: v["rank"] for k, v in COMPOUND_META.items()
        }

        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "F1FeatureTransformer":
        """
        Learn fold-wise statistics from training data only.

        If PitNextLap is present in X it is used to calibrate compound cliff
        laps from the actual pit distribution, then dropped. It must never
        survive into transform() output.

        y is accepted but ignored internally - the target is read from X directly
        when present, so it is available for cliff calibration during fit().
        """
        df = X.copy()

        steps = [
            "race total laps",
            "race mean lap time (SC-filtered)",
            "compound cliff laps [TARGET-DERIVED]",
            "pit cluster map     [TARGET-ADJACENT]",
            "label encoders",
        ]

        with tqdm(steps, desc="fit()", disable=not self.verbose, leave=False) as pbar:

            # 1. Max lap number per race tells us how many laps the race has total
            pbar.set_description("fit() › race total laps")
            for (race, year), grp in df.groupby(["Race", "Year"]):
                self._race_total_laps[(race, year)] = grp["LapNumber"].max()
            pbar.update(1)

            # 2. Mean lap time per race, excluding safety car laps (>130s are SC/VSC)
            pbar.set_description("fit() › race mean lap time")
            clean = df[df["LapTime (s)"] < 130]
            for (race, year), grp in clean.groupby(["Race", "Year"]):
                self._race_mean_laptime[(race, year)] = grp["LapTime (s)"].mean()
            pbar.update(1)

            # 3. Compound cliff laps: use the 75th percentile TyreLife at actual pit rows.
            #    This is target-derived so it must only ever run on training data.
            pbar.set_description("fit() › compound cliff [TARGET-DERIVED]")
            if _TARGET_COL in df.columns:
                pit_rows = df[df[_TARGET_COL] == 1.0]
                for comp, grp in pit_rows.groupby("Compound"):
                    q75 = grp["TyreLife"].quantile(0.75)
                    if not np.isnan(q75):
                        self._compound_cliff[comp] = int(q75)
            pbar.update(1)

            # 4. How many drivers pitted on each lap of each race.
            #    so transform() can do a fast O(1) lookup. The original used
            #    MultiIndex.map() which breaks on a default RangeIndex.
            pbar.set_description("fit() › pit cluster map [TARGET-ADJACENT]")
            cluster_series = (
                df.groupby(["Race", "Year", "LapNumber"])["PitStop"]
                .sum()
            )
            self._pit_cluster_map = {
                (race, year, int(lap)): int(count)
                for (race, year, lap), count in cluster_series.items()
            }
            pbar.update(1)
            # 5. LabelEncoders for driver and race string IDs
            pbar.set_description("fit() › label encoders")
            self._driver_enc.fit(df["Driver"].astype(str))
            self._race_enc.fit(df["Race"].astype(str))
            pbar.update(1)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Build the engineered feature matrix.

        The target column PitNextLap is always stripped from the output,
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        df = X.copy()

        # leak into the output feature matrix
        if _TARGET_COL in df.columns:
            df = df.drop(columns=[_TARGET_COL])
        df = df.sort_values(_GROUP + ["LapNumber"]).reset_index(drop=True)

        steps = [
            "A · Tyre degradation        [ROW-SAFE]",
            "B · Lap time performance    [FOLD-WISE: laptime_zscore]",
            "C · Rolling / lag features  [ROW-SAFE]",
            "D · Race strategy & pos     [FOLD-WISE: laps_remaining]",
            "E · Compound one-hot        [ROW-SAFE]",
            "F · Race phase & stint      [ROW-SAFE]",
            "G · Key interactions        [ROW-SAFE]",
            "H · Safety-car proxy        [FOLD-WISE*: pit_cluster + zscore]",
            "I · Encodings               [FOLD-WISE]",
            "  · Cleanup",
        ]

        with tqdm(steps, desc="transform()", disable=not self.verbose, leave=False) as pbar:

            # A. TYRE DEGRADATION - all pure arithmetic on row values, no leakage risk
            pbar.set_description(steps[0])
            df["tyre_life_sq"]  = df["TyreLife"] ** 2
            df["tyre_life_log"] = np.log1p(df["TyreLife"])

            df["compound_expected_stint"] = (
                df["Compound"].map(self._compound_expected).fillna(17)
            )
            df["tyre_life_norm"]      = df["TyreLife"] / df["compound_expected_stint"]
            df["tyre_overrun_amount"] = (df["tyre_life_norm"] - 1.0).clip(lower=0)

            df["compound_cliff_lap"] = df["Compound"].map(self._compound_cliff).fillna(20)
            df["laps_to_cliff"]      = (df["compound_cliff_lap"] - df["TyreLife"]).clip(lower=-20)
            df["past_cliff"]         = (df["TyreLife"] > df["compound_cliff_lap"]).astype(int)
            # Exponential proximity score: 1.0 when right at the cliff, decays as you get further away
            df["cliff_proximity"]    = np.exp(-0.15 * df["laps_to_cliff"].clip(lower=0))

            df["deg_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1)
            df["deg_accel"]   = df["Cumulative_Degradation"] / (df["TyreLife"].clip(lower=1) ** 2)

            df["compound_softness"] = (
                df["Compound"].map(self._compound_softness).fillna(3)
            )
            df["is_wet_compound"]  = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
            df["is_soft_compound"] = (df["Compound"] == "SOFT").astype(int)
            pbar.update(1)

            # B. LAP TIME PERFORMANCE - uses race mean learned in fit(), so fold-wise
            pbar.set_description(steps[1])
            race_mean_lt = df.apply(
                lambda r: self._race_mean_laptime.get(
                    (r["Race"], r["Year"]), r["LapTime (s)"]
                ),
                axis=1,
            )
            df["laptime_zscore"]    = (df["LapTime (s)"] - race_mean_lt) / (race_mean_lt * 0.05 + 1e-6)
            df["laptime_delta_abs"] = df["LapTime_Delta"].abs()
            df["laptime_worsening"] = (df["LapTime_Delta"] > 0.25).astype(int)
            pbar.update(1)

            # C. ROLLING / LAG FEATURES
            # not by value. Without the default argument trick (_w=w), all rolling features
            # would use the last value of w and be identical. The _w=w freezes each lambda
            # to its own window size at definition time.
            pbar.set_description(steps[2])
            grp_obj = df.groupby(_GROUP)
            for w in tqdm(
                self.lag_windows,
                desc="  rolling windows",
                disable=not self.verbose,
                leave=False,
            ):
                df[f"roll_lt_mean_{w}"] = grp_obj["LapTime (s)"].transform(
                    lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean()
                )
                df[f"roll_lt_std_{w}"]  = grp_obj["LapTime (s)"].transform(
                    lambda s, _w=w: s.shift(1).rolling(_w, min_periods=min(2, _w)).std().fillna(0)
                )
                df[f"roll_delta_{w}"]   = grp_obj["LapTime_Delta"].transform(
                    lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean()
                )
                df[f"roll_deg_{w}"]     = grp_obj["Cumulative_Degradation"].transform(
                    lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean()
                )

            df["laptime_lag1"] = grp_obj["LapTime (s)"].transform(lambda s: s.shift(1))

            def _pace_momentum(s: pd.Series) -> pd.Series:
                # Slope of a 3-lap rolling window of lap times - positive means slowing down
                return s.shift(1).rolling(3, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
                    raw=True,
                )

            df["pace_momentum_3"] = grp_obj["LapTime (s)"].transform(_pace_momentum)

            # the outer group context, giving wrong run-length counts when rows from
            # multiple (Race, Year, Driver) groups were interleaved.
            # Correct approach: within each group, find where the worsening flag changes
            # direction using cumsum, then cumsum again within each run.
            def _consecutive_worse(s: pd.Series) -> pd.Series:
                flag      = (s > 0.25).astype(float)
                run_id    = (flag != flag.shift()).cumsum()
                run_cumsum = flag.groupby(run_id).cumsum()
                return run_cumsum
            df["consecutive_worse"] = grp_obj["LapTime_Delta"].transform(_consecutive_worse)
            df["prev_pitstop"]      = grp_obj["PitStop"].transform(lambda s: s.shift(1).fillna(0))
            pbar.update(1)

            # D. RACE STRATEGY & POSITION - uses race total laps learned in fit()
            pbar.set_description(steps[3])
            df["total_race_laps"] = df.apply(
                lambda r: self._race_total_laps.get((r["Race"], r["Year"]), 70.0),
                axis=1,
            )
            df["laps_remaining"]      = df["total_race_laps"] - df["LapNumber"]
            df["laps_remaining_norm"] = df["laps_remaining"] / df["total_race_laps"]
            df["too_late_to_pit"]     = (df["laps_remaining"] <= 5).astype(int)
            df["undercut_window"]     = (
                (df["laps_remaining"] > 10) & (df["tyre_life_norm"] > 0.7)
            ).astype(int)
            df["overcut_candidate"]   = (
                (df["tyre_life_norm"] > 1.0) & (df["laptime_zscore"] < 0.5)
            ).astype(int)

            df["position_norm"]       = df["Position"] / 20.0
            df["position_change_abs"] = df["Position_Change"].abs()
            df["in_points"]           = (df["Position"] <= 10).astype(int)
            df["roll_pos_mean_3"]     = grp_obj["Position"].transform(
                lambda s: s.shift(1).rolling(3, min_periods=1).mean()
            )
            df["position_trend"]   = df["Position"] - df["roll_pos_mean_3"]
            df["stint_exhaustion"] = df["tyre_life_norm"] * df["compound_softness"]
            pbar.update(1)

            # E. COMPOUND ONE-HOT - row-safe, no leakage risk
            pbar.set_description(steps[4])
            cpd_dummies = pd.get_dummies(df["Compound"], prefix="cpd", dtype=int)
            # Ensure all 5 compound columns exist even if some compounds never appear
            for col in _COMPOUND_COLS:
                if col not in cpd_dummies.columns:
                    cpd_dummies[col] = 0
            df = pd.concat([df, cpd_dummies[_COMPOUND_COLS]], axis=1)
            df["cpd_rank_x_life"] = df["compound_softness"] * df["tyre_life_norm"]
            pbar.update(1)

            # F. RACE PHASE & STINT - row-safe, thresholds are fixed constants
            pbar.set_description(steps[5])
            df["race_phase_early"] = (df["RaceProgress"] < 0.33).astype(int)
            df["race_phase_mid"]   = (
                (df["RaceProgress"] >= 0.33) & (df["RaceProgress"] < 0.67)
            ).astype(int)
            df["race_phase_late"]  = (df["RaceProgress"] >= 0.67).astype(int)
            df["in_pit_window"]    = (
                (df["RaceProgress"] > 0.25) & (df["RaceProgress"] < 0.82)
            ).astype(int)
            df["is_first_stint"]     = (df["Stint"] == 1).astype(int)
            df["laps_past_expected"] = (
                df["TyreLife"] - df["compound_expected_stint"]
            ).clip(lower=0)
            pbar.update(1)

            # G. KEY INTERACTIONS - multiplicative combinations of existing features
            pbar.set_description(steps[6])
            df["tyreage_x_progress"] = df["tyre_life_norm"] * df["RaceProgress"]
            df["urgency_index"]      = (
                df["tyre_life_norm"] / df["laps_remaining_norm"].clip(lower=0.01)
            )
            df["cliff_x_window"]     = df["cliff_proximity"] * df["in_pit_window"]
            df["momentum_x_overrun"] = (
                df["pace_momentum_3"].fillna(0)
                * (df["tyre_overrun_amount"] > 0).astype(int)
            )
            pbar.update(1)

            # H. SAFETY-CAR / CLUSTER PROXY
            # sc_lap_proxy flags any lap where the z-score is high enough to suggest SC pace.
            # pit_cluster_size uses the dict learned in fit() to count how many drivers
            # pitted on the same lap - a spike often signals a SC period.
            pbar.set_description(steps[7])
            df["sc_lap_proxy"]  = (df["laptime_zscore"] > 0.8).astype(int)
            df["sc_spike_flag"] = (df["laptime_zscore"] > 0.6).astype(float)

            # then do a dict lookup. The original MultiIndex.map() approach silently
            # returns all-NaN when df has a default RangeIndex after reset_index().
            pit_key = list(zip(
                df["Race"],
                df["Year"],
                df["LapNumber"].astype(int),
            ))
            df["pit_cluster_size"] = pd.array(
                [self._pit_cluster_map.get(k, 0) for k in pit_key],
                dtype="float64",
            )
            df["is_cluster_lap"]  = (df["pit_cluster_size"] >= 3).astype(int)
            df["recent_sc_proxy"] = (
                df.groupby(_GROUP)["sc_spike_flag"]
                .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
                .fillna(0)
            )
            pbar.update(1)

            # I. ENCODINGS - converts string Driver and Race IDs to integers.
            # Unseen values at inference time get -1 so the model can handle them.
            pbar.set_description(steps[8])
            known_drivers = set(self._driver_enc.classes_)
            known_races   = set(self._race_enc.classes_)
            df["driver_enc"] = df["Driver"].astype(str).apply(
                lambda d: self._driver_enc.transform([d])[0] if d in known_drivers else -1
            )
            df["race_enc"] = df["Race"].astype(str).apply(
                lambda r: self._race_enc.transform([r])[0] if r in known_races else -1
            )
            pbar.update(1)

            # CLEANUP: drop raw string columns and ensure no NaNs remain
            pbar.set_description(steps[9])
            cols_to_drop = ["Driver", "Race", "Compound"] if self.drop_raw_ids else []
            # Always drop the target column regardless of drop_raw_ids
            cols_to_drop.append(_TARGET_COL)
            df = df.drop(
                columns=[c for c in cols_to_drop if c in df.columns]
            )
            df = df.ffill().fillna(0)
            pbar.update(1)

        return df

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kw
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Expected output column names after transform(), in output order."""
        rolling_cols = [
            f"{prefix}_{w}"
            for prefix in ["roll_lt_mean", "roll_lt_std", "roll_delta", "roll_deg"]
            for w in self.lag_windows
        ]
        return [
            "Year", "PitStop", "LapNumber", "Stint", "TyreLife",
            "Position", "LapTime (s)", "LapTime_Delta",
            "Cumulative_Degradation", "RaceProgress", "Position_Change",
            "tyre_life_sq", "tyre_life_log",
            "compound_expected_stint", "tyre_life_norm", "tyre_overrun_amount",
            "compound_cliff_lap", "laps_to_cliff", "past_cliff", "cliff_proximity",
            "deg_per_lap", "deg_accel",
            "compound_softness", "is_wet_compound", "is_soft_compound",
            "laptime_zscore", "laptime_delta_abs", "laptime_worsening",
            *rolling_cols,
            "laptime_lag1", "pace_momentum_3", "consecutive_worse", "prev_pitstop",
            "total_race_laps", "laps_remaining", "laps_remaining_norm",
            "too_late_to_pit", "undercut_window", "overcut_candidate",
            "position_norm", "position_change_abs", "in_points",
            "roll_pos_mean_3", "position_trend", "stint_exhaustion",
            *_COMPOUND_COLS, "cpd_rank_x_life",
            "race_phase_early", "race_phase_mid", "race_phase_late",
            "in_pit_window", "is_first_stint", "laps_past_expected",
            "tyreage_x_progress", "urgency_index", "cliff_x_window", "momentum_x_overrun",
            "sc_lap_proxy", "pit_cluster_size", "is_cluster_lap",
            "sc_spike_flag", "recent_sc_proxy",
            "driver_enc", "race_enc",
        ]

class F1FoldWiseWrapper:
    """
    Runs fold-safe feature engineering across a full CV loop.

    Ensures fit() is always called on training data and transform() on the
    corresponding validation data, never the other way around.

    Parameters
    ----------
    n_splits : int
        Number of CV folds (default 5).
    random_state : int
        Seed for StratifiedKFold reproducibility.
    lag_windows : list[int]
        Passed to F1FeatureTransformer.
    drop_raw_ids : bool
        Passed to F1FeatureTransformer.
    verbose : bool
        Passed to F1FeatureTransformer.

    like lag_window instead of lag_windows. Now uses explicit keyword arguments
    so any mistyped parameter raises a TypeError immediately.
    """
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        lag_windows: Optional[list[int]] = None,
        drop_raw_ids: bool = True,
        verbose: bool = True,
    ) -> None:
        self.n_splits     = n_splits
        self.random_state = random_state
        # Keep transformer kwargs in a dict for clean passing and easy inspection
        self._transformer_kwargs = dict(
            lag_windows  = lag_windows or [1, 3, 5],
            drop_raw_ids = drop_raw_ids,
            verbose      = verbose,
        )

        self._fold_transformers: list[F1FeatureTransformer] = []
        self._full_transformer:  Optional[F1FeatureTransformer] = None
        self._skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    def fit_transform_fold(
        self,
        X_train_fold: pd.DataFrame,
        y_train_fold: pd.Series,
        X_val_fold: pd.DataFrame,
        fold_index: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit a fresh transformer on one training split and transform both splits.

        to the raw training frame. This matters because sort_values() inside
        transform() rearranges rows, so pre-attaching the target would misalign it.
        Returns: (X_tr_feat, X_val_feat) - engineered feature matrices with no
        target column and no leakage from validation data.
        """
        # the target already attached. Better to fail loudly than produce wrong results.
        if _TARGET_COL in X_train_fold.columns:
            raise ValueError(
                f"'{_TARGET_COL}' found in X_train_fold before fit_transform_fold(). "
                "Pass the raw feature frame without the target column; "
                "fit_transform_fold() attaches y internally for cliff calibration."
            )
        if _TARGET_COL in X_val_fold.columns:
            raise ValueError(
                f"'{_TARGET_COL}' found in X_val_fold. "
                "Validation frames must never contain the target column."
            )
        tqdm.write(f"\n── Fold {fold_index + 1} / {self.n_splits} " + "─" * 40)
        transformer = F1FeatureTransformer(**self._transformer_kwargs)

        # Temporarily attach the target so fit() can calibrate compound cliff laps
        X_tr_with_target = X_train_fold.copy()
        X_tr_with_target[_TARGET_COL] = y_train_fold.values

        with tqdm(total=2, desc=f"  Fold {fold_index + 1}",
                  disable=not self._transformer_kwargs["verbose"]) as pbar:
            pbar.set_description(f"  Fold {fold_index + 1} › fit on train split")
            transformer.fit(X_tr_with_target, y_train_fold)
            pbar.update(1)

            pbar.set_description(f"  Fold {fold_index + 1} › transform train + val")
            # Pass the original frames (without target). transform() will also strip
            # the target if it somehow ends up there - second safety layer.
            X_tr_feat  = transformer.transform(X_train_fold)
            X_val_feat = transformer.transform(X_val_fold)
            pbar.update(1)

        self._fold_transformers.append(transformer)
        return X_tr_feat, X_val_feat

    def fit_full(self, X_full: pd.DataFrame, y_full: pd.Series) -> "F1FoldWiseWrapper":
        """
        Fit one transformer on the entire training set for test-set inference.
        Never use the output of this for OOF evaluation - that would be leakage.
        """
        if _TARGET_COL in X_full.columns:
            raise ValueError(
                f"'{_TARGET_COL}' found in X_full. Pass the raw feature frame."
            )
        tqdm.write("\n── Full-data fit (test inference transformer) " + "─" * 27)
        self._full_transformer = F1FeatureTransformer(**self._transformer_kwargs)
        X_with_target = X_full.copy()
        X_with_target[_TARGET_COL] = y_full.values
        self._full_transformer.fit(X_with_target, y_full)
        return self

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Transform the test set using the full-data transformer. Call fit_full() first."""
        if self._full_transformer is None:
            raise RuntimeError("Call fit_full() before transform_test().")
        tqdm.write("\n── Transforming test set " + "─" * 47)
        return self._full_transformer.transform(X_test)

    def run_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[np.ndarray], list[np.ndarray]]:
        """
        Run the full fold-wise feature engineering loop and return all results.

        Returns:
          train_feats : engineered training DataFrames, one per fold
          val_feats   : engineered validation DataFrames, one per fold
          train_idxs  : integer row index arrays for each training split
          val_idxs    : integer row index arrays for each validation split
        """
        train_feats, val_feats = [], []
        train_idxs,  val_idxs  = [], []

        tqdm.write(
            f"\n{'═' * 62}\n"
            f"  F1FoldWiseWrapper — {self.n_splits}-fold stratified CV\n"
            f"  Rows: {len(X):,}  |  Positive rate: {y.mean():.1%}\n"
            f"{'═' * 62}"
        )

        fold_iter = tqdm(
            enumerate(self._skf.split(X, y)),
            total=self.n_splits,
            desc="CV folds",
        )

        for fold, (tr_idx, val_idx) in fold_iter:
            X_tr_feat, X_val_feat = self.fit_transform_fold(
                X.iloc[tr_idx], y.iloc[tr_idx],
                X.iloc[val_idx], fold_index=fold,
            )
            train_feats.append(X_tr_feat)
            val_feats.append(X_val_feat)
            train_idxs.append(tr_idx)
            val_idxs.append(val_idx)

        tqdm.write(f"\n{'─' * 62}\n✓  All {self.n_splits} folds complete.\n{'─' * 62}\n")
        return train_feats, val_feats, train_idxs, val_idxs
