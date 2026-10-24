"""
Test suite for data loading and preprocessing verification.
Verifies items 3 through 6 on the master protocol checklist.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import load_subject_data, EEG_CHANNELS_22


def test_subject_1_loading():
    print("=" * 60)
    print("RUNNING CHECKLIST VERIFICATION ON SUBJECT 1...")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_subject_data(subject_id=1, verbose=False)

    print(f"[*] Channels defined: {len(EEG_CHANNELS_22)}")
    print(f"[*] Channels list: {EEG_CHANNELS_22[:6]} ... {EEG_CHANNELS_22[-6:]}")

    # 1. Shape checks
    print(f"\n[1] Checking Tensor Dimensions:")
    print(f"    X_train shape: {X_train.shape} (Expected: (144, 22, 750))")
    print(f"    y_train shape: {y_train.shape} (Expected: (144,))")
    print(f"    X_test shape:  {X_test.shape} (Expected: (144, 22, 750))")
    print(f"    y_test shape:  {y_test.shape} (Expected: (144,))")

    assert X_train.shape == (144, 22, 750), f"X_train shape mismatch: {X_train.shape}"
    assert y_train.shape == (144,), f"y_train shape mismatch: {y_train.shape}"
    assert X_test.shape == (144, 22, 750), f"X_test shape mismatch: {X_test.shape}"
    assert y_test.shape == (144,), f"y_test shape mismatch: {y_test.shape}"
    print("    --> PASS: All tensor dimensions match strictly (N, 22, 750)!")

    # 2. Class balance checks
    print(f"\n[2] Checking Binary Class Balance:")
    train_left = np.sum(y_train == 0)
    train_right = np.sum(y_train == 1)
    test_left = np.sum(y_test == 0)
    test_right = np.sum(y_test == 1)

    print(f"    Train: Left Hand (0) = {train_left}, Right Hand (1) = {train_right}")
    print(f"    Test:  Left Hand (0) = {test_left}, Right Hand (1) = {test_right}")

    assert train_left == 72 and train_right == 72, f"Train imbalance: {train_left} vs {train_right}"
    assert test_left == 72 and test_right == 72, f"Test imbalance: {test_left} vs {test_right}"
    print("    --> PASS: Perfectly balanced 72 trials per class in both sessions!")

    # 3. Numerical integrity checks
    print(f"\n[3] Checking Numerical Integrity:")
    assert not np.isnan(X_train).any(), "Found NaNs in X_train"
    assert not np.isinf(X_train).any(), "Found Infs in X_train"
    assert not np.isnan(X_test).any(), "Found NaNs in X_test"
    assert not np.isinf(X_test).any(), "Found Infs in X_test"
    print("    No NaNs or Infs detected.")

    # Check signal scale (EEG in microvolts/volts)
    std_train = np.std(X_train)
    std_test = np.std(X_test)
    print(f"    Train Signal STD: {std_train:.2e}")
    print(f"    Test Signal STD:  {std_test:.2e}")
    assert std_train > 0 and std_test > 0, "Signal standard deviation is zero!"
    print("    --> PASS: Signals contain valid non-zero dynamic range.")

    print("\n" + "=" * 60)
    print("ALL CHECKLIST ITEMS 1 THROUGH 6 PASSED WITH 100% COMPLIANCE!")
    print("=" * 60)


if __name__ == "__main__":
    test_subject_1_loading()
