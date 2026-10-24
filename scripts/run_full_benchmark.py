"""
Full 9-Subject Benchmark Runner.

Frozen Protocol:
- All 9 subjects (A01 to A09).
- Persisted stratified 44-trial test split per subject (configs/splits/).
- 100-trial calibration pool per subject.
- 5 calibration seeds [0, 1, 2, 3, 4] with nested budgets [10, 25, 50, 100].
- Evaluates 5 conditions for both CSP+LDA and Riemannian Tangent Space.
- Saves raw results, summary tables, and generates publication plots.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne

# Suppress verbose logging
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import load_subject_data
from evaluation import get_or_create_fixed_target_split, evaluate_five_conditions


def run_full_benchmark():
    print("=" * 90)
    print("STARTING FULL BENCHMARK ACROSS ALL 9 SUBJECTS (FROZEN PROTOCOL)")
    print("=" * 90)

    subjects = list(range(1, 10))
    budgets = [10, 25, 50, 100]
    seeds = [0, 1, 2, 3, 4]
    models = [("CSP + LDA", "csp_lda"), ("Riemannian Tangent Space", "riemannian")]
    conditions = ["baseline_s1", "target_only", "naive_pooling", "ea_unsupervised", "ea_supervised"]

    records = []

    for subj in subjects:
        print(f"\n[*] Processing Subject {subj:02d}/09...")
        X_train, y_train, X_test_all, y_test_all = load_subject_data(subj, verbose=False)

        # 1. Get/create persisted stratified target split
        target_split = get_or_create_fixed_target_split(
            subject_id=subj,
            X_target_all=X_test_all,
            y_target_all=y_test_all,
            test_size=44,
            random_state=2026,
            splits_dir="configs/splits"
        )

        X_cal_pool = target_split["X_cal_pool"]
        y_cal_pool = target_split["y_cal_pool"]
        X_test_held_out = target_split["X_test"]
        y_test_held_out = target_split["y_test"]

        # 2. Iterate over models
        for model_label, model_type in models:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                permuted_indices = rng.permutation(len(X_cal_pool))

                for b in budgets:
                    selected_idx = permuted_indices[:b]
                    X_cal = X_cal_pool[selected_idx]
                    y_cal = y_cal_pool[selected_idx]

                    eval_out = evaluate_five_conditions(
                        X_train=X_train,
                        y_train=y_train,
                        X_cal=X_cal,
                        y_cal=y_cal,
                        X_test=X_test_held_out,
                        y_test=y_test_held_out,
                        model_type=model_type
                    )

                    for cond in conditions:
                        records.append({
                            "subject": f"S{subj:02d}",
                            "model": model_label,
                            "budget": b,
                            "seed": seed,
                            "condition": cond,
                            "balanced_accuracy": eval_out["scores"][cond]
                        })

    # Convert to DataFrame
    df_raw = pd.DataFrame(records)

    # Output directory
    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save raw seed-level data
    raw_path = tables_dir / "benchmark_raw_results.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"\n[+] Saved raw seed-level runs to {raw_path}")

    # Compute summary table (mean and std across seeds)
    df_summary = df_raw.groupby(["subject", "model", "budget", "condition"])["balanced_accuracy"].agg(["mean", "std"]).reset_index()
    summary_path = tables_dir / "benchmark_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[+] Saved summary table to {summary_path}")

    # Compute Cohort-Level Average Table (mean across all 9 subjects)
    df_cohort = df_summary.groupby(["model", "budget", "condition"])["mean"].agg(["mean", "std"]).reset_index()
    df_cohort.rename(columns={"mean": "cohort_mean", "std": "cohort_std"}, inplace=True)
    cohort_path = tables_dir / "benchmark_cohort_summary.csv"
    df_cohort.to_csv(cohort_path, index=False)
    print(f"[+] Saved cohort summary to {cohort_path}")

    # Print Cohort Summary Table to Console
    print("\n" + "=" * 90)
    print("COHORT-LEVEL BENCHMARK RESULTS (AVERAGE ACROSS ALL 9 SUBJECTS)")
    print("=" * 90)

    for model_label, _ in models:
        print(f"\nPIPELINE: {model_label} (Balanced Accuracy %)")
        print(f"{'Budget':<8} | {'Baseline S1':<16} | {'Target-Only':<16} | {'Naive Pooling':<16} | {'EA (Unsup)':<16} | {'EA (Supervised)':<16}")
        print("-" * 96)

        df_m = df_cohort[df_cohort["model"] == model_label]
        for b in budgets:
            row = f"{b:<8} | "
            for cond in conditions:
                sub = df_m[(df_m["budget"] == b) & (df_m["condition"] == cond)]
                if len(sub) > 0 and not np.isnan(sub["cohort_mean"].values[0]):
                    m = sub["cohort_mean"].values[0] * 100
                    s = sub["cohort_std"].values[0] * 100
                    row += f"{m:5.1f}% +/- {s:4.1f}% | "
                else:
                    row += f"{'N/A':<16} | "
            print(row)
        print("-" * 96)

    # Generate Publication Figures
    print("\n[*] Generating Publication Figures...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Calibration Learning Curves (Figure 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    palette = {
        "baseline_s1": "#7f7f7f",
        "target_only": "#1f77b4",
        "naive_pooling": "#ff7f0e",
        "ea_unsupervised": "#2ca02c",
        "ea_supervised": "#d62728"
    }
    labels_map = {
        "baseline_s1": "Baseline S1 (Zero-Shot)",
        "target_only": "Target-Only (No Transfer)",
        "naive_pooling": "Naive Pooling (S1 + Cal)",
        "ea_unsupervised": "EA Unsupervised (0 Labels)",
        "ea_supervised": "EA Supervised (S1 + Cal)"
    }

    for idx, (m_label, _) in enumerate(models):
        ax = axes[idx]
        df_plot = df_raw[df_raw["model"] == m_label].copy()
        df_plot["condition_label"] = df_plot["condition"].map(labels_map)

        sns.lineplot(
            data=df_plot,
            x="budget",
            y="balanced_accuracy",
            hue="condition",
            palette=palette,
            marker="o",
            err_style="bars",
            errorbar="se",
            ax=ax
        )
        ax.set_title(f"{m_label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Target Calibration Budget (Trials)", fontsize=12)
        ax.set_ylabel("Balanced Accuracy" if idx == 0 else "", fontsize=12)
        ax.set_ylim(0.45, 1.0)
        ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="Random Guess (50%)")
        ax.legend(title="Condition", loc="lower right", fontsize=9)

    plt.suptitle("Calibration Learning Curves: Cross-Session Adaptation vs Budgets", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig1_path = figures_dir / "fig1_calibration_learning_curves.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[+] Saved Figure 1 to {fig1_path}")

    # 2. Subject-Wise Adaptation Delta Chart (Figure 2) for Budget 25
    b_focus = 25
    df_b25 = df_summary[df_summary["budget"] == b_focus].copy()
    piv = df_b25.pivot(index=["subject", "model"], columns="condition", values="mean").reset_index()
    piv["delta_ea_unsup"] = (piv["ea_unsupervised"] - piv["baseline_s1"]) * 100
    piv["delta_ea_sup"] = (piv["ea_supervised"] - piv["baseline_s1"]) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    df_delta = piv[piv["model"] == "CSP + LDA"].copy()
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in df_delta["delta_ea_unsup"]]

    bars = ax.bar(df_delta["subject"], df_delta["delta_ea_unsup"], color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title(f"Subject-Wise Adaptation Delta (Δs = EA Unsupervised - Baseline S1) at Budget = {b_focus} Trials (CSP+LDA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("Balanced Accuracy Gain Δs (%)", fontsize=12)

    for bar, val in zip(bars, df_delta["delta_ea_unsup"]):
        y_pos = val + 0.8 if val >= 0 else val - 1.8
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:+.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig2_path = figures_dir / "fig2_subject_delta_bars.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[+] Saved Figure 2 to {fig2_path}")

    print("\n" + "=" * 90)
    print("FULL 9-SUBJECT BENCHMARK SUCCESSFULLY COMPLETED!")
    print("=" * 90)


if __name__ == "__main__":
    run_full_benchmark()
