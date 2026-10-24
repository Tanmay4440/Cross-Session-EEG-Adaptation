"""
Evaluation framework for cross-session motor imagery BCI experiments.

Strictly locks in the 5 Cardinal Rules:
1. One persisted, stratified 44-trial test split per subject (exactly 22 Left, 22 Right).
2. The remaining 100 trials form the calibration pool.
3. Random seeds control ONLY calibration-subset sampling, using nested permutations across budgets.
4. Held-out test set is completely frozen across all conditions, budgets, and seeds.
5. Baseline S1 evaluates on the frozen test set and has zero variance (0.0%) across calibration seeds.
6. Target alignment (EA Unsupervised) receives only unlabelled X_cal.
7. Saves all predictions, confusion matrices, and balanced accuracy scores.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

from alignment import EuclideanAlignment
from models import get_csp_lda_pipeline, get_riemannian_pipeline


def get_or_create_fixed_target_split(
    subject_id: int,
    X_target_all: np.ndarray,
    y_target_all: np.ndarray,
    test_size: int = 44,
    random_state: int = 2026,
    splits_dir: str = "configs/splits"
) -> Dict[str, Any]:
    """
    Create or load a persisted, stratified target-session split once per subject.

    Guarantees:
    - Exactly `test_size // 2` trials per class in the held-out test set (22 Left, 22 Right).
    - Exactly 100 trials in the calibration pool (50 Left, 50 Right).
    - Indices are sorted and persisted to disk so all future runs evaluate on
      the exact same physical trials.
    """
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)
    split_file = splits_path / f"subject_{subject_id:02d}_split.npz"

    if split_file.exists():
        data = np.load(str(split_file))
        cal_pool_idx = data["cal_pool_idx"]
        test_idx = data["test_idx"]
    else:
        indices = np.arange(len(y_target_all))
        cal_pool_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y_target_all
        )
        cal_pool_idx = np.sort(cal_pool_idx)
        test_idx = np.sort(test_idx)
        np.savez_compressed(str(split_file), cal_pool_idx=cal_pool_idx, test_idx=test_idx)

    # Verify disjointness
    assert len(set(cal_pool_idx).intersection(set(test_idx))) == 0, "Leakage: Overlapping split indices!"

    X_cal_pool = X_target_all[cal_pool_idx]
    y_cal_pool = y_target_all[cal_pool_idx]
    X_test = X_target_all[test_idx]
    y_test = y_target_all[test_idx]

    return {
        "cal_pool_idx": cal_pool_idx,
        "test_idx": test_idx,
        "X_cal_pool": X_cal_pool,
        "y_cal_pool": y_cal_pool,
        "X_test": X_test,
        "y_test": y_test,
    }


def sample_calibration_nested(
    n_pool: int,
    cal_budget: int,
    seed: int
) -> np.ndarray:
    """
    Sample calibration indices using a single random permutation per seed.

    Ensures nested budgets: budget 10 is indices[:10], budget 25 is indices[:25],
    budget 50 is indices[:50], and budget 100 is indices[:100].
    """
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n_pool)
    return permuted[:cal_budget]


def evaluate_five_conditions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "csp_lda"
) -> Dict[str, Any]:
    """
    Evaluate all 5 experimental conditions on the exact same fixed held-out test set (X_test, y_test).

    Returns
    -------
    dict with:
      - 'scores': dict mapping condition -> balanced_accuracy
      - 'predictions': dict mapping condition -> y_pred
      - 'confusion_matrices': dict mapping condition -> confusion_matrix
    """
    def _create_model():
        if model_type == "csp_lda":
            return get_csp_lda_pipeline(n_components=4)
        elif model_type == "riemannian":
            return get_riemannian_pipeline(cov_estimator="lwf", C=1.0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    scores = {}
    predictions = {}
    matrices = {}

    def _record(cond_name, y_pred):
        ba = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        scores[cond_name] = ba
        predictions[cond_name] = y_pred
        matrices[cond_name] = cm

    # 1. Condition 1: Baseline S1 (Trained on Day 1 only, no adaptation, zero Day 2 data)
    clf_base = _create_model()
    clf_base.fit(X_train, y_train)
    _record("baseline_s1", clf_base.predict(X_test))

    # 2. Condition 2: Target-Only (Trained on Day 2 calibration budget only, zero Day 1 data)
    if len(np.unique(y_cal)) > 1 and len(y_cal) >= 10:
        try:
            clf_tgt = _create_model()
            clf_tgt.fit(X_cal, y_cal)
            _record("target_only", clf_tgt.predict(X_test))
        except Exception:
            scores["target_only"] = np.nan
            predictions["target_only"] = None
            matrices["target_only"] = None
    else:
        scores["target_only"] = np.nan
        predictions["target_only"] = None
        matrices["target_only"] = None

    # 3. Condition 3: Naive Pooling (Day 1 + Day 2 calibration dumped together without alignment)
    if len(X_cal) > 0:
        X_pooled = np.concatenate([X_train, X_cal], axis=0)
        y_pooled = np.concatenate([y_train, y_cal], axis=0)
        clf_naive = _create_model()
        clf_naive.fit(X_pooled, y_pooled)
        _record("naive_pooling", clf_naive.predict(X_test))
    else:
        _record("naive_pooling", predictions["baseline_s1"])

    # 4. Condition 4 & 5: Euclidean Alignment
    # Source alignment (fitted on Day 1 only)
    ea_src = EuclideanAlignment()
    X_train_ea = ea_src.fit_transform(X_train)

    # Target alignment (STRICTLY INDUCTIVE: fitted on unlabelled X_cal only, zero labels passed!)
    ea_tgt = EuclideanAlignment()
    ea_tgt.fit(X_cal)
    X_cal_ea = ea_tgt.transform(X_cal)
    X_test_ea = ea_tgt.transform(X_test)

    # 4. Condition 4: EA Unsupervised (Trained on aligned Day 1 only; ZERO Day 2 labels used)
    clf_ea_unsup = _create_model()
    clf_ea_unsup.fit(X_train_ea, y_train)
    _record("ea_unsupervised", clf_ea_unsup.predict(X_test_ea))

    # 5. Condition 5: EA Supervised (Trained on aligned Day 1 + aligned Day 2 calibration data)
    X_ea_pooled = np.concatenate([X_train_ea, X_cal_ea], axis=0)
    y_ea_pooled = np.concatenate([y_train, y_cal], axis=0)
    clf_ea_sup = _create_model()
    clf_ea_sup.fit(X_ea_pooled, y_ea_pooled)
    _record("ea_supervised", clf_ea_sup.predict(X_test_ea))

    return {
        "scores": scores,
        "predictions": predictions,
        "confusion_matrices": matrices
    }
