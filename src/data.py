"""
Data loader and preprocessing module for BCI Competition IV Dataset 2a.

Strictly adheres to master_experimental_protocol.md:
- Extracts exactly the 22 EEG channels (drops auxiliary EOG channels).
- Standardizes channel names to the international 10-20 system.
- Applies zero-phase 4th-order Butterworth bandpass filter (8.0 Hz - 30.0 Hz).
- Epochs 3-second motor imagery windows [1.0s, 4.0s] post-cue -> exactly 750 samples at 250 Hz.
- Binary motor imagery: Left Hand (Class 1, y=0) vs Right Hand (Class 2, y=1).
- Loads true ground-truth labels for Session E from official true_labels/ mat files.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import scipy.io
import mne

# Official 22 EEG channel layout for BCI Competition IV 2a
EEG_CHANNELS_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "Oz"
]

# Event codes in BCI Competition IV 2a
EVENT_CUE_LEFT = "769"     # Left Hand
EVENT_CUE_RIGHT = "770"    # Right Hand
EVENT_CUE_FOOT = "771"     # Foot
EVENT_CUE_TONGUE = "772"   # Tongue
EVENT_CUE_MASKED = "783"   # Masked evaluation cue


def load_raw_continuous(
    subject_id: int,
    session: str = "T",
    data_dir: str = "official data/BCICIV_2a_gdf",
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    verbose: bool = False
) -> mne.io.Raw:
    """
    Load raw continuous EEG, select 22 channels, apply montage, and bandpass filter.

    Parameters
    ----------
    subject_id : int (1 to 9)
    session : str ("T" for training session, "E" for evaluation session)
    data_dir : str path to the folder containing GDF files
    l_freq : float lower cutoff frequency in Hz (default 8.0)
    h_freq : float higher cutoff frequency in Hz (default 30.0)
    verbose : bool whether to print MNE logging

    Returns
    -------
    raw : mne.io.Raw preprocessed continuous EEG
    """
    session = session.upper()
    if session not in ("T", "E"):
        raise ValueError(f"Session must be 'T' or 'E', got {session}")
    if not (1 <= subject_id <= 9):
        raise ValueError(f"subject_id must be between 1 and 9, got {subject_id}")

    filename = f"A{subject_id:02d}{session}.gdf"
    filepath = Path(data_dir) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"GDF file not found: {filepath}")

    # Read raw GDF file
    raw = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=verbose)

    # The first 22 channels are EEG, remaining 3 are EOG
    eeg_raw_names = raw.ch_names[:22]
    raw.pick(eeg_raw_names, verbose=verbose)

    # Rename channels to standard 10-20 system
    rename_dict = dict(zip(eeg_raw_names, EEG_CHANNELS_22))
    raw.rename_channels(rename_dict, verbose=verbose)
    raw.set_channel_types({ch: "eeg" for ch in EEG_CHANNELS_22}, verbose=verbose)

    # Assign standard 10-20 montage for spatial topographic plotting
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=verbose)

    # Apply 4th-order zero-phase Butterworth bandpass filter
    iir_params = dict(order=4, ftype="butter")
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=iir_params,
        verbose=verbose
    )

    return raw


def extract_epochs(
    raw: mne.io.Raw,
    subject_id: int,
    session: str = "T",
    tmin: float = 1.0,
    tmax: float = 4.0,
    true_labels_dir: str = "official data/true_labels",
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 3-second motor imagery epochs for binary classification (Left vs Right).

    Parameters
    ----------
    raw : mne.io.Raw preprocessed continuous EEG
    subject_id : int (1 to 9)
    session : str ("T" or "E")
    tmin : float start time in seconds relative to cue (default 1.0)
    tmax : float end time in seconds relative to cue (default 4.0)
    true_labels_dir : str path to true label .mat files
    verbose : bool whether to print MNE logging

    Returns
    -------
    X : np.ndarray shape (n_trials, 22, 750)
    y : np.ndarray shape (n_trials,) with binary labels {0: Left Hand, 1: Right Hand}
    """
    session = session.upper()
    events, event_id = mne.events_from_annotations(raw, verbose=verbose)
    sfreq = raw.info["sfreq"]  # 250 Hz

    # Adjusted tmax so that (tmax - tmin) * sfreq is exactly 750 samples (3.0 seconds)
    n_samples_expected = int(round((tmax - tmin) * sfreq))
    tmax_adjusted = tmin + (n_samples_expected - 1) / sfreq

    if session == "T":
        # In Session T, class cues 769 (Left) and 770 (Right) are present in the GDF
        if EVENT_CUE_LEFT not in event_id or EVENT_CUE_RIGHT not in event_id:
            raise KeyError(f"Expected event cues {EVENT_CUE_LEFT} and {EVENT_CUE_RIGHT} not found in Session T annotations.")

        selected_event_ids = {
            "left": event_id[EVENT_CUE_LEFT],
            "right": event_id[EVENT_CUE_RIGHT]
        }

        epochs = mne.Epochs(
            raw,
            events,
            event_id=selected_event_ids,
            tmin=tmin,
            tmax=tmax_adjusted,
            baseline=None,
            preload=True,
            verbose=verbose
        )

        X = epochs.get_data(copy=True)  # shape: (n_trials, 22, 750)
        # Map event codes: 'left' -> 0, 'right' -> 1
        raw_events = epochs.events[:, 2]
        y = np.zeros(len(raw_events), dtype=int)
        y[raw_events == selected_event_ids["right"]] = 1

    else:
        # In Session E, cues are masked with code 783 (EVENT_CUE_MASKED)
        if EVENT_CUE_MASKED not in event_id:
            raise KeyError(f"Masked event cue {EVENT_CUE_MASKED} not found in Session E annotations.")

        masked_event_id = {"masked": event_id[EVENT_CUE_MASKED]}
        epochs = mne.Epochs(
            raw,
            events,
            event_id=masked_event_id,
            tmin=tmin,
            tmax=tmax_adjusted,
            baseline=None,
            preload=True,
            verbose=verbose
        )

        X_all = epochs.get_data(copy=True)  # all 288 trials

        # Load official ground-truth labels
        mat_path = Path(true_labels_dir) / f"A{subject_id:02d}E.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"True labels file not found: {mat_path}")

        mat_data = scipy.io.loadmat(str(mat_path))
        classlabels = mat_data["classlabel"].flatten()  # 288 labels (1=Left, 2=Right, 3=Foot, 4=Tongue)

        if len(classlabels) != len(X_all):
            raise ValueError(f"Mismatch: found {len(X_all)} epochs in GDF but {len(classlabels)} labels in MAT file.")

        # Filter strictly for binary classes: 1 (Left) and 2 (Right)
        binary_mask = np.isin(classlabels, [1, 2])
        X = X_all[binary_mask]
        # Map to {0: Left Hand, 1: Right Hand}
        y = (classlabels[binary_mask] - 1).astype(int)

    return X, y


def load_subject_data(
    subject_id: int,
    data_dir: str = "official data/BCICIV_2a_gdf",
    true_labels_dir: str = "official data/true_labels",
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete end-to-end loader for a single subject returning training (T) and evaluation (E) sets.

    Returns
    -------
    X_train : np.ndarray shape (144, 22, 750)
    y_train : np.ndarray shape (144,) labels {0, 1}
    X_test  : np.ndarray shape (144, 22, 750)
    y_test  : np.ndarray shape (144,) labels {0, 1}
    """
    # Load and epoch Session T
    raw_t = load_raw_continuous(subject_id, session="T", data_dir=data_dir, verbose=verbose)
    X_train, y_train = extract_epochs(raw_t, subject_id, session="T", true_labels_dir=true_labels_dir, verbose=verbose)

    # Load and epoch Session E
    raw_e = load_raw_continuous(subject_id, session="E", data_dir=data_dir, verbose=verbose)
    X_test, y_test = extract_epochs(raw_e, subject_id, session="E", true_labels_dir=true_labels_dir, verbose=verbose)

    return X_train, y_train, X_test, y_test
