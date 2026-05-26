"""
F1 Pit Stop Prediction - Consensus Feature Selector

This file handles feature selection after preprocessing. It does four things:
  1. Removes columns that are constant (zero variance) and carry no information.
  2. Computes feature importance from three different angles: model-based (GBDT
     split gain), permutation (how much AUC drops when the feature is shuffled),
     and SHAP (mean absolute contribution across training rows).
  3. Aggregates all three into one consensus ranking so no single method dominates.
  4. Runs a batch forward selection: starts with the top N features and tries
     adding more in batches, keeping a batch only if it measurably improves
     cross-validated AUC and the improvement is stable across inner folds.

All of this runs strictly inside fit(), which must be called on training-fold
rows only. Never call fit() on the full dataset or on validation data.

"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Optional heavy dependencies - gracefully skip if not installed
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

class _ConstantColumnFilter(BaseEstimator, TransformerMixin):
    """
    Removes columns that never change (variance == 0).

    threshold=0.0 is intentional: we only remove truly constant columns,
    not low-variance ones. Rare binary indicators like is_wet_compound are
    still informative even if they're mostly zero.
    """

    def __init__(self) -> None:
        self._vt = VarianceThreshold(threshold=0.0)

    def fit(self, X: pd.DataFrame, y=None) -> "_ConstantColumnFilter":
        self._cols_in = list(X.columns)
        self._vt.fit(X)
        self._cols_out: list[str] = [
            col for col, keep in zip(self._cols_in, self._vt.get_support())
            if keep
        ]
        removed = [c for c in self._cols_in if c not in self._cols_out]
        if removed:
            logger.info("ConstantColumnFilter removed %d column(s): %s", len(removed), removed)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self._cols_out].copy()

    @property
    def kept_columns(self) -> list[str]:
        return self._cols_out

class _ImportanceComputer:
    """
    Computes three importance scores for each feature using a fitted model:
      1. model importance  - native GBDT split-gain (fast, but can be biased)
      2. permutation       - how much AUC drops when this feature is shuffled
      3. shap              - mean |SHAP value| across a sample of training rows

    All three are computed on training-fold data only, then normalized to [0, 1]
    so they can be combined on a common scale in the consensus ranker.
    """

    def __init__(
        self,
        estimator,
        n_repeats: int = 10,
        shap_sample: int = 5000,
        random_state: int = 42,
    ) -> None:
        self.estimator    = estimator
        self.n_repeats    = n_repeats
        self.shap_sample  = shap_sample
        self.random_state = random_state

    def compute(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> dict[str, pd.Series]:
        """
        Returns a dict of importance Series indexed by feature name.
        Keys: "model", "permutation", "shap" (shap may be absent if not installed).
        """
        cols  = list(X_train.columns)
        X_arr = X_train.values
        y_arr = y_train.values
        results: dict[str, pd.Series] = {}

        # 1. Model-based importance from the fitted GBDT
        model = clone(self.estimator)
        model.fit(X_arr, y_arr)

        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        else:
            fi = np.zeros(len(cols))
            logger.warning(
                "Estimator has no feature_importances_ attribute; "
                "model importance will be zero."
            )
        results["model"] = pd.Series(fi, index=cols)

        # 2. Permutation importance: shuffle each feature and measure the AUC drop.
        #    Clipped at 0 because a negative permutation importance just means the
        #    feature is slightly worse than noise - treat it as zero.
        perm = permutation_importance(
            model, X_arr, y_arr,
            n_repeats=self.n_repeats,
            scoring="roc_auc",
            random_state=self.random_state,
            n_jobs=-1,
        )
        results["permutation"] = pd.Series(
            perm.importances_mean.clip(min=0), index=cols
        )

        # 3. SHAP importance: mean absolute SHAP value across a subsample of rows.
        #    for binary classification. Newer shap returns a single ndarray of shape
        #    (n_samples, n_features, 2). Handle both.
        if _HAS_SHAP:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_arr), size=min(self.shap_sample, len(X_arr)), replace=False)
            X_shap = X_arr[idx]
            try:
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)

                # Old shap API: returns list [neg_class, pos_class]
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                # New shap API: returns ndarray of shape (n, features, 2)
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    shap_values = shap_values[:, :, 1]

                shap_imp = np.abs(shap_values).mean(axis=0)
                results["shap"] = pd.Series(shap_imp, index=cols)
            except Exception as exc:
                logger.warning("SHAP computation failed (%s); skipping.", exc)
        else:
            logger.info("shap not installed; SHAP importance skipped.")

        return results

class _ConsensusRanker:
    """
    Combines multiple importance vectors into one consensus ranking.

    Algorithm:
      For each importance method m and feature f:
        1. Normalize importance to [0, 1]:  imp_norm = imp / max(imp)
        2. Compute rank position among all features
        3. Normalize rank to [0, 1]:        rank_norm = rank / n_features
        4. Consensus score = mean of rank_norm across all methods
        5. Sort descending - higher score means more important

    Using rank rather than raw importance avoids any one method's scale
    dominating the others.
    """

    def aggregate(
        self,
        importance_dict: dict[str, pd.Series],
    ) -> pd.Series:
        """Returns a consensus score Series. Higher score = more important feature."""
        if not importance_dict:
            raise ValueError("importance_dict is empty.")

        frames: list[pd.Series] = []
        all_features = list(next(iter(importance_dict.values())).index)
        n = len(all_features)

        for method, imp in importance_dict.items():
            imp       = imp.reindex(all_features).fillna(0)
            max_val   = imp.max()
            imp_norm  = imp / (max_val + 1e-12)
            rank      = imp_norm.rank(ascending=False, method="average")
            rank_norm = rank / n
            frames.append(rank_norm.rename(method))

        df_ranks        = pd.concat(frames, axis=1)
        # 1 - mean_rank so that rank 1 (best feature) gets a high score
        consensus_score = 1.0 - df_ranks.mean(axis=1)
        return consensus_score.sort_values(ascending=False)

class _BatchForwardSelector:
    """
    Iterative batch-forward feature selection driven by cross-validated ROC-AUC.

    Evaluates batches of B features at a time rather than one by one. This
    captures interaction effects and avoids prematurely discarding features that
    look weak in isolation but are useful in combination.

    A batch is only accepted if:
      - it improves AUC by at least min_auc_gain
      - the AUC standard deviation across inner folds is below stability_std_threshold
        (i.e. the improvement is consistent, not driven by one lucky fold)
    """

    def __init__(
        self,
        estimator,
        top_n_start: int = 40,
        batch_size: int = 5,
        min_auc_gain: float = 0.0005,
        stability_std_threshold: float = 0.005,
        n_inner_splits: int = 3,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.estimator               = estimator
        self.top_n_start             = top_n_start
        self.batch_size              = batch_size
        self.min_auc_gain            = min_auc_gain
        self.stability_std_threshold = stability_std_threshold
        self.n_inner_splits          = n_inner_splits
        self.random_state            = random_state
        self.verbose                 = verbose
        self._log: list[dict] = []

    def _cv_auc(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
    ) -> tuple[float, float]:
        """Cross-validate AUC on the given feature subset. Returns (mean, std)."""
        skf = StratifiedKFold(
            n_splits=self.n_inner_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = cross_val_score(
            clone(self.estimator),
            X[features].values,
            y.values,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return float(scores.mean()), float(scores.std())

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ranked_features: list[str],
    ) -> list[str]:
        """
        Run batch-forward selection and return the final feature list.

        Parameters
        ----------
        X               : training-fold features (preprocessed, no target column)
        y               : training-fold binary target
        ranked_features : feature names in consensus rank order, best first
        """
        self._log = []

        # Only use features that actually exist in X - ranked_features may include
        # columns that were dropped by the constant filter
        ranked_available = [f for f in ranked_features if f in X.columns]

        if len(ranked_available) == 0:
            logger.warning("No ranked features present in X — returning empty selection.")
            return []
        # Warm start: begin with the top N features and measure their baseline AUC
        warm_n     = min(self.top_n_start, len(ranked_available))
        selected   = list(ranked_available[:warm_n])
        candidates = list(ranked_available[warm_n:])

        base_auc, base_std = self._cv_auc(X, y, selected)
        if self.verbose:
            logger.info(
                "Warm-start: %d features | AUC = %.5f ± %.5f",
                len(selected), base_auc, base_std,
            )
        self._log.append({
            "iteration": 0,
            "action":     "warm_start",
            "n_features": len(selected),
            "auc_mean":   base_auc,
            "auc_std":    base_std,
            "batch":      [],
            "gain":       0.0,
            "stable":     True,
        })

        if not candidates:
            if self.verbose:
                logger.info(
                    "All %d available features consumed by warm start — "
                    "no batch iterations to run.",
                    len(selected),
                )
            return selected
        iteration = 1
        while candidates:
            batch = candidates[: self.batch_size]

            # not from the already-accepted `selected` list. The original code
            # would break the while loop if any batch feature was missing,
            # which would silently discard all remaining candidates.
            valid_batch   = [f for f in batch if f in X.columns]
            missing_batch = [f for f in batch if f not in X.columns]
            if missing_batch:
                logger.warning(
                    "Iter %d: skipping %d batch feature(s) not in X: %s",
                    iteration, len(missing_batch), missing_batch,
                )
            if not valid_batch:
                # The entire batch was missing from X - skip it and move on
                candidates = candidates[self.batch_size:]
                iteration += 1
                continue

            proposed = selected + valid_batch
            cand_auc, cand_std = self._cv_auc(X, y, proposed)
            gain   = cand_auc - base_auc
            stable = cand_std <= self.stability_std_threshold

            if gain >= self.min_auc_gain and stable:
                action   = "accept"
                selected = proposed
                base_auc = cand_auc
                base_std = cand_std
                if self.verbose:
                    logger.info(
                        "Iter %3d ACCEPT batch %s | AUC %.5f (+%.5f) std %.5f",
                        iteration, valid_batch, cand_auc, gain, cand_std,
                    )
            else:
                action = "reject"
                reason = []
                if gain < self.min_auc_gain:
                    reason.append(f"gain {gain:+.5f} < threshold {self.min_auc_gain}")
                if not stable:
                    reason.append(f"std {cand_std:.5f} > threshold {self.stability_std_threshold}")
                if self.verbose:
                    logger.info(
                        "Iter %3d REJECT batch %s | AUC %.5f (%s)",
                        iteration, valid_batch, cand_auc, "; ".join(reason),
                    )

            self._log.append({
                "iteration":  iteration,
                "action":     action,
                "batch":      valid_batch,
                "n_features": len(selected),
                "auc_mean":   cand_auc,
                "auc_std":    cand_std,
                "gain":       gain,
                "stable":     stable,
            })

            candidates = candidates[self.batch_size:]
            iteration += 1

        if self.verbose:
            logger.info(
                "Forward selection complete: %d features selected | final AUC %.5f",
                len(selected), base_auc,
            )

        return selected

    @property
    def selection_log(self) -> pd.DataFrame:
        """Per-iteration decision log as a DataFrame, useful for debugging."""
        return pd.DataFrame(self._log)

class ConsensusFeatureSelector(BaseEstimator, TransformerMixin):
    """
    End-to-end feature selection pipeline for binary classification.

    Steps run inside fit():
      1. Remove fully constant columns (VarianceThreshold with threshold=0).
      2. Fit the estimator on the non-constant training features.
      3. Compute three importance vectors: model, permutation, SHAP.
      4. Aggregate into a consensus rank ordering.
      5. Run batch-forward selection with inner-CV AUC gate and stability check.

    Parameters
    ----------
    estimator : sklearn-compatible classifier
        LGBMClassifier recommended. Defaults to a sensible LightGBM config.
    top_n_start : int
        How many top-ranked features to start with in warm start (default 40).
    batch_size : int
        How many candidate features to evaluate per forward-selection iteration.
    min_auc_gain : float
        Minimum AUC improvement needed to accept a batch.
    stability_std_threshold : float
        Maximum fold-AUC std allowed to accept a batch (stability gate).
    n_inner_splits : int
        Number of inner CV folds for batch evaluation.
    perm_n_repeats : int
        How many times to repeat each permutation importance shuffle.
    shap_sample : int
        Number of training rows to subsample for SHAP computation.
    random_state : int
    verbose : bool

    Attributes after fit():
      selected_features_  : list[str]      - the final selected feature names
      consensus_ranking_  : pd.Series      - all features ranked by consensus score
      importance_table_   : pd.DataFrame   - raw scores from all three methods
      selection_log_      : pd.DataFrame   - per-iteration forward selection decisions
      removed_constant_   : list[str]      - columns removed by variance filter
    """

    def __init__(
        self,
        estimator=None,
        top_n_start: int = 40,
        batch_size: int = 5,
        min_auc_gain: float = 0.0005,
        stability_std_threshold: float = 0.005,
        n_inner_splits: int = 3,
        perm_n_repeats: int = 10,
        shap_sample: int = 5000,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.estimator               = estimator
        self.top_n_start             = top_n_start
        self.batch_size              = batch_size
        self.min_auc_gain            = min_auc_gain
        self.stability_std_threshold = stability_std_threshold
        self.n_inner_splits          = n_inner_splits
        self.perm_n_repeats          = perm_n_repeats
        self.shap_sample             = shap_sample
        self.random_state            = random_state
        self.verbose                 = verbose

    def _get_estimator(self):
        """Return the user-provided estimator or a sensible LightGBM default."""
        if self.estimator is not None:
            return clone(self.estimator)
        if _HAS_LGB:
            return lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=20,
                random_state=self.random_state,
                verbose=-1,
            )
        raise ImportError(
            "LightGBM is not installed and no estimator was provided. "
            "Install lightgbm or pass an explicit estimator."
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConsensusFeatureSelector":
        """
        Run the full feature-selection pipeline on training-fold data.
        Must only be called on the training split inside a CV fold.
        """
        # Stage 1: Drop constant columns
        _log("Stage 1: VarianceThreshold — removing constant columns")
        const_filter = _ConstantColumnFilter()
        const_filter.fit(X, y)
        X_nconst = const_filter.transform(X)

        self.removed_constant_ = [
            c for c in X.columns if c not in const_filter.kept_columns
        ]
        _log(f"  Removed {len(self.removed_constant_)} constant column(s).")

        # Stage 2: Compute importance from all three methods.
        # clone() calls inside _ImportanceComputer and _BatchForwardSelector
        # always work from the canonical source, not an already-cloned copy.
        _log("Stage 2: Computing multi-method feature importances")
        imp_computer = _ImportanceComputer(
            estimator   = self.estimator if self.estimator is not None else self._get_estimator(),
            n_repeats   = self.perm_n_repeats,
            shap_sample = self.shap_sample,
            random_state= self.random_state,
        )
        importance_dict = imp_computer.compute(X_nconst, y)
        self.importance_table_ = pd.DataFrame(importance_dict).fillna(0)
        _log(f"  Importance methods computed: {list(importance_dict.keys())}")

        # Stage 3: Aggregate into a single consensus ranking
        _log("Stage 3: Consensus rank aggregation")
        ranker = _ConsensusRanker()
        self.consensus_ranking_ = ranker.aggregate(importance_dict)
        ranked_features = list(self.consensus_ranking_.index)
        _log(f"  Top 10 consensus features: {ranked_features[:10]}")

        # Stage 4: Batch-forward selection with inner-CV AUC gate
        _log(
            f"Stage 4: Batch-forward selection "
            f"(top_n_start={self.top_n_start}, batch_size={self.batch_size}, "
            f"min_gain={self.min_auc_gain})"
        )
        fwd_selector = _BatchForwardSelector(
            estimator               = self.estimator if self.estimator is not None else self._get_estimator(),
            top_n_start             = self.top_n_start,
            batch_size              = self.batch_size,
            min_auc_gain            = self.min_auc_gain,
            stability_std_threshold = self.stability_std_threshold,
            n_inner_splits          = self.n_inner_splits,
            random_state            = self.random_state,
            verbose                 = self.verbose,
        )
        self.selected_features_ = fwd_selector.select(X_nconst, y, ranked_features)
        self.selection_log_     = fwd_selector.selection_log
        _log(
            f"  Final selection: {len(self.selected_features_)} features "
            f"(from {X_nconst.shape[1]} non-constant)"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only the selected features. Safe to call on train or val/test data.
        Missing features get a warning and are skipped rather than raising.
        """
        check_is_fitted(self, "selected_features_")
        present = [c for c in self.selected_features_ if c in X.columns]
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            logger.warning(
                "ConsensusFeatureSelector.transform: %d selected feature(s) "
                "not found in X and will be skipped: %s",
                len(missing), missing,
            )
        return X[present].copy()

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kw
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def selection_report(self) -> str:
        """Human-readable summary of what the selection pipeline did."""
        check_is_fitted(self, "selected_features_")
        lines = [
            "=" * 68,
            "ConsensusFeatureSelector — Selection Report",
            "=" * 68,
            f"Constant columns removed  : {len(self.removed_constant_)}",
            f"  {self.removed_constant_}",
            "",
            "Consensus top-10 features (pre-forward-selection):",
        ]
        for rank, (feat, score) in enumerate(self.consensus_ranking_.head(10).items(), 1):
            lines.append(f"  {rank:>2}. {feat:<40s}  score={score:.4f}")

        lines += [
            "",
            f"Forward-selection log ({len(self.selection_log_)} iterations):",
        ]
        for _, row in self.selection_log_.iterrows():
            if row["iteration"] == 0:
                lines.append(
                    f"  Iter  0  warm-start  n={row['n_features']}  "
                    f"AUC={row['auc_mean']:.5f} ± {row['auc_std']:.5f}"
                )
            else:
                flag = "✓ ACCEPT" if row["action"] == "accept" else "✗ REJECT"
                lines.append(
                    f"  Iter {int(row['iteration']):>2}  {flag}  "
                    f"batch={row['batch']}  "
                    f"gain={row.get('gain', 0):+.5f}  "
                    f"AUC={row['auc_mean']:.5f} ± {row['auc_std']:.5f}"
                )

        lines += [
            "",
            f"Final selected features: {len(self.selected_features_)}",
            "  " + ", ".join(self.selected_features_),
            "=" * 68,
        ]
        return "\n".join(lines)

    def importance_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame with per-method importances and consensus score,
        sorted from most to least important, with a column showing which
        features made the final cut.
        """
        check_is_fitted(self, "consensus_ranking_")
        df = self.importance_table_.copy()
        df["consensus_score"] = self.consensus_ranking_.reindex(df.index).fillna(0)
        df["selected"]        = df.index.isin(self.selected_features_)
        return df.sort_values("consensus_score", ascending=False)

class FoldSafeSelectionWrapper:
    """
    Runs ConsensusFeatureSelector across a full CV loop and tracks results.

    Parameters
    ----------
    selector_kwargs : dict
        Passed to ConsensusFeatureSelector.
    n_splits : int
    random_state : int
    """

    def __init__(
        self,
        selector_kwargs: Optional[dict] = None,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.selector_kwargs = selector_kwargs or {}
        self.n_splits        = n_splits
        self.random_state    = random_state

        self._fold_selectors: list[ConsensusFeatureSelector] = []
        self._full_selector:  Optional[ConsensusFeatureSelector] = None

    def fit_transform_fold(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        fold_index: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, ConsensusFeatureSelector]:
        """
        Fit a fresh selector on one training split, then apply it to both splits.

        Returns (X_tr_selected, X_val_selected, fitted_selector).
        The selector is also stored internally for later analysis.
        """
        logger.info("Fold %d — fitting ConsensusFeatureSelector", fold_index + 1)
        selector = ConsensusFeatureSelector(**self.selector_kwargs)
        selector.fit(X_tr, y_tr)
        self._fold_selectors.append(selector)
        return selector.transform(X_tr), selector.transform(X_val), selector

    def fit_full(
        self, X_full: pd.DataFrame, y_full: pd.Series
    ) -> "FoldSafeSelectionWrapper":
        """Fit on the entire training set. Used only for test-set inference."""
        logger.info("Full-data fit — ConsensusFeatureSelector")
        self._full_selector = ConsensusFeatureSelector(**self.selector_kwargs)
        self._full_selector.fit(X_full, y_full)
        return self

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Apply the full-data selector to the test set. Call fit_full() first."""
        if self._full_selector is None:
            raise RuntimeError("Call fit_full() before transform_test().")
        return self._full_selector.transform(X_test)

    def consensus_feature_overlap(self) -> pd.DataFrame:
        """
        Shows how consistently each feature was selected across folds.

        selection_rate column isn't silently computed over the wrong denominator.
        Returns a DataFrame with columns:
          feature, n_folds_selected, selection_rate, mean_consensus_score
        """
        if not self._fold_selectors:
            raise RuntimeError("No folds have been run yet.")

        n_run = len(self._fold_selectors)
        if n_run != self.n_splits:
            logger.warning(
                "consensus_feature_overlap(): n_splits=%d but only %d fold(s) "
                "have been run. selection_rate is computed over %d fold(s).",
                self.n_splits, n_run, n_run,
            )

        # Collect every feature that appeared in any fold's ranking or selection
        all_features: set[str] = set()
        for sel in self._fold_selectors:
            all_features.update(sel.selected_features_)
            all_features.update(sel.consensus_ranking_.index.tolist())

        rows = []
        for feat in sorted(all_features):
            n_selected = sum(
                1 for sel in self._fold_selectors if feat in sel.selected_features_
            )
            # Series.get() can return a Series (not a scalar) if the index has
            # duplicates, which would break np.mean() silently.
            scores = [
                float(sel.consensus_ranking_.reindex([feat]).fillna(0.0).iloc[0])
                for sel in self._fold_selectors
            ]
            rows.append({
                "feature":              feat,
                "n_folds_selected":     n_selected,
                "selection_rate":       n_selected / n_run,
                "mean_consensus_score": float(np.mean(scores)),
            })
        return (
            pd.DataFrame(rows)
            .sort_values(["n_folds_selected", "mean_consensus_score"], ascending=False)
            .reset_index(drop=True)
        )

def _log(msg: str) -> None:
    """Log at INFO level and also print to stdout."""
    logger.info(msg)
    print(msg)
