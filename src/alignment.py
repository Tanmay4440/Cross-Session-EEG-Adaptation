"""
Euclidean Alignment (EA) transformer for EEG data.

Based on He & Wu (2019):
"Euclidean Space Data Alignment for Cross-Subject and Cross-Session EEG-Based Brain-Computer Interfaces"

Strictly inductive implementation:
- Input shape: (n_trials, n_channels, n_times)
- Computes reference covariance R_bar = (1/N) * sum( (1/T) * X_i @ X_i.T )
- Uses scale-invariant relative regularization: R_reg = R_bar + (reg_eps * Tr(R_bar)/C) * I
  This guarantees numerical stability even when EEG voltages are in microvolts (1e-6)
  and when N < n_channels (e.g., N=10 calibration budget).
- Computes R_bar^(-1/2) via eigendecomposition.
- Transforms trials: X_aligned = R_bar^(-1/2) @ X
- After alignment, the session's mean covariance matrix is the Identity Matrix I.
"""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EuclideanAlignment(BaseEstimator, TransformerMixin):
    """
    Euclidean Alignment for multi-channel EEG epochs.

    Parameters
    ----------
    reg_eps : float
        Relative regularization fraction added to the diagonal:
        R_reg = R_bar + (reg_eps * trace(R_bar) / C) * I.
        Scale-invariant and numerically safe for any calibration budget N.
    """

    def __init__(self, reg_eps: float = 1e-5):
        self.reg_eps = reg_eps
        self.R_ref_: Optional[np.ndarray] = None
        self.R_inv_sqrt_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "EuclideanAlignment":
        """
        Estimate the reference covariance matrix and its inverse square root.

        Parameters
        ----------
        X : np.ndarray of shape (n_trials, n_channels, n_times)
        y : ignored (unsupervised alignment)

        Returns
        -------
        self : EuclideanAlignment
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (n_trials, n_channels, n_times), got shape {X.shape}")

        n_trials, n_channels, n_times = X.shape

        # Compute empirical covariance for each trial: (1/n_times) * X_i @ X_i.T
        # Shape: (n_trials, n_channels, n_channels)
        trial_covs = np.matmul(X, X.transpose(0, 2, 1)) / n_times

        # Reference covariance: arithmetic mean across all trials
        R_bar = np.mean(trial_covs, axis=0)

        # Scale-invariant regularization based on average channel power
        trace_val = np.trace(R_bar)
        mean_power = trace_val / n_channels if trace_val > 0 else 1.0
        reg_amount = self.reg_eps * mean_power

        R_bar_reg = R_bar + reg_amount * np.eye(n_channels)

        # Compute R^(-1/2) via symmetric eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R_bar_reg)

        # Ensure all eigenvalues are positive (clipped to relative min)
        min_allowed = max(reg_amount * 0.1, 1e-18)
        eigenvalues = np.clip(eigenvalues, a_min=min_allowed, a_max=None)
        inv_sqrt_diag = 1.0 / np.sqrt(eigenvalues)

        # R^(-1/2) = V @ diag(1/sqrt(lambda)) @ V.T
        R_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_diag) @ eigenvectors.T

        # Symmetrize to prevent tiny numerical asymmetry
        R_inv_sqrt = 0.5 * (R_inv_sqrt + R_inv_sqrt.T)

        self.R_ref_ = R_bar_reg
        self.R_inv_sqrt_ = R_inv_sqrt
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Align EEG epochs using the fitted R^(-1/2) whitening transformation.

        Parameters
        ----------
        X : np.ndarray of shape (n_trials, n_channels, n_times)

        Returns
        -------
        X_aligned : np.ndarray of shape (n_trials, n_channels, n_times)
        """
        if self.R_inv_sqrt_ is None:
            raise RuntimeError("EuclideanAlignment must be fitted before calling transform.")
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (n_trials, n_channels, n_times), got shape {X.shape}")

        # X_aligned = R_inv_sqrt @ X for each trial
        # R_inv_sqrt is (C, C), X is (N, C, T) -> output is (N, C, T)
        X_aligned = np.matmul(self.R_inv_sqrt_, X)
        return X_aligned

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to X, then transform X."""
        return self.fit(X, y).transform(X)
