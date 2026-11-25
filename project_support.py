"""Reusable, pickle-safe transformers and grouped stacking for NoFlyZone."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression


class SurveyFeatures(TransformerMixin, BaseEstimator):
    """Row-local aggregates; never uses the label or another passenger's values."""
    def __init__(self, rating_columns=(), enabled=True):
        self.rating_columns = rating_columns
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        if self.enabled:
            ratings = result.loc[:, list(self.rating_columns)]
            count = ratings.notna().sum(axis=1)
            result['rating_mean'] = ratings.mean(axis=1)
            result['rating_count'] = count
            result['low_rating_fraction'] = ratings.le(2).sum(axis=1).div(count.replace(0, np.nan))
        return result


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Fit winsorization bounds on the training partition only; preserve NaNs."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_ = np.nanquantile(values, self.upper, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class GroupedStackingClassifier(ClassifierMixin, BaseEstimator):
    """Out-of-fold meta-features with each base pipeline fitted inside its fold.

    This avoids ordinary StackingClassifier's internal row-wise CV, which could
    separate identical feature profiles. Outer CV routes groups explicitly.
    """
    def __init__(self, estimators, n_splits=3, random_state=42):
        self.estimators = estimators
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError('Grouped stacking requires profile groups.')
        cv = list(StratifiedGroupKFold(self.n_splits, shuffle=True,
                  random_state=self.random_state).split(X, y, groups))
        columns = []
        self.estimators_ = []
        for name, estimator in self.estimators:
            oof = cross_val_predict(clone(estimator), X, y, cv=cv,
                                    method='predict_proba', n_jobs=1)[:, 1]
            columns.append(oof)
            self.estimators_.append((name, clone(estimator).fit(X, y)))
        self.final_estimator_ = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.final_estimator_.fit(np.column_stack(columns), y)
        self.classes_ = self.final_estimator_.classes_
        return self

    def predict_proba(self, X):
        Z = np.column_stack([m.predict_proba(X)[:, 1] for _, m in self.estimators_])
        return self.final_estimator_.predict_proba(Z)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
