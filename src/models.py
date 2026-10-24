"""
Model definitions for motor imagery BCI.

Pipelines:
1. CSP + LDA: Conventional spatial filtering baseline.
2. Covariances + TangentSpace + LogisticRegression: Riemannian geometry baseline.
"""

from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def get_csp_lda_pipeline(n_components: int = 4, reg: Optional[float] = None) -> Pipeline:
    """
    Construct a Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) pipeline.

    Parameters
    ----------
    n_components : int
        Number of spatial filters (typically 4: 2 top and 2 bottom).
    reg : float or None
        Optional covariance shrinkage regularization for CSP.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    """
    csp = CSP(
        n_components=n_components,
        reg=reg,
        log=True,
        norm_trace=False
    )
    lda = LDA(solver="svd")
    return Pipeline([("csp", csp), ("lda", lda)])


def get_riemannian_pipeline(
    cov_estimator: str = "lwf",
    C: float = 1.0,
    metric: str = "riemann"
) -> Pipeline:
    """
    Construct a Riemannian Tangent Space classification pipeline.

    Covariances -> TangentSpace projection -> Logistic Regression.

    Parameters
    ----------
    cov_estimator : str
        Covariance estimator (default 'lwf' for Ledoit-Wolf shrinkage).
    C : float
        Inverse regularization strength for Logistic Regression.
    metric : str
        Riemannian metric for Tangent Space reference (default 'riemann').

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    """
    cov = Covariances(estimator=cov_estimator)
    ts = TangentSpace(metric=metric)
    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
    return Pipeline([("cov", cov), ("ts", ts), ("clf", clf)])
