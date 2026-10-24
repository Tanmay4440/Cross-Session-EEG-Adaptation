"""
Generate a clean, straightforward PDF table containing all benchmark results
for all 9 subjects across both pipelines (CSP+LDA and Riemannian Tangent Space).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_pdf_results():
    csv_path = Path("results/tables/benchmark_summary.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    pdf_path = Path("results/all_9_subjects_results.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        models = [("CSP + LDA Pipeline", "CSP + LDA"), ("Riemannian Tangent Space Pipeline", "Riemannian Tangent Space")]

        for title, model_name in models:
            df_m = df[df["model"] == model_name].copy()

            # For each budget (10, 25, 50, 100), build a clean table for all 9 subjects
            for budget in [10, 25, 50, 100]:
                df_b = df_m[df_m["budget"] == budget]
                piv = df_b.pivot(index="subject", columns="condition", values="mean")

                # Format rows
                table_data = []
                headers = ["Subject", "Baseline S1", "Target-Only", "Naive Pooling", "EA (Unsup)", "EA (Sup)", "Delta (EA - Base)"]

                for subj in sorted(piv.index):
                    row = [subj]
                    b_s1 = piv.loc[subj, "baseline_s1"] * 100
                    tgt = piv.loc[subj, "target_only"] * 100 if "target_only" in piv.columns and not np.isnan(piv.loc[subj, "target_only"]) else None
                    naive = piv.loc[subj, "naive_pooling"] * 100
                    ea_unsup = piv.loc[subj, "ea_unsupervised"] * 100
                    ea_sup = piv.loc[subj, "ea_supervised"] * 100
                    delta = ea_unsup - b_s1

                    row.append(f"{b_s1:.1f}%")
                    row.append(f"{tgt:.1f}%" if tgt is not None else "N/A")
                    row.append(f"{naive:.1f}%")
                    row.append(f"{ea_unsup:.1f}%")
                    row.append(f"{ea_sup:.1f}%")
                    row.append(f"{delta:+.1f}%")
                    table_data.append(row)

                # Add Cohort Average row
                avg_row = ["AVERAGE"]
                avg_base = piv["baseline_s1"].mean() * 100
                avg_tgt = piv["target_only"].mean() * 100 if "target_only" in piv.columns else None
                avg_naive = piv["naive_pooling"].mean() * 100
                avg_ea_unsup = piv["ea_unsupervised"].mean() * 100
                avg_ea_sup = piv["ea_supervised"].mean() * 100
                avg_delta = avg_ea_unsup - avg_base

                avg_row.append(f"{avg_base:.1f}%")
                avg_row.append(f"{avg_tgt:.1f}%" if avg_tgt is not None else "N/A")
                avg_row.append(f"{avg_naive:.1f}%")
                avg_row.append(f"{avg_ea_unsup:.1f}%")
                avg_row.append(f"{avg_ea_sup:.1f}%")
                avg_row.append(f"{avg_delta:+.1f}%")
                table_data.append(avg_row)

                # Render table on matplotlib figure
                fig, ax = plt.subplots(figsize=(11, 6.5))
                ax.axis("tight")
                ax.axis("off")

                fig.suptitle(f"{title} | Calibration Budget = {budget} Trials", fontsize=14, fontweight="bold", y=0.96)

                table = ax.table(
                    cellText=table_data,
                    colLabels=headers,
                    cellLoc="center",
                    loc="center"
                )

                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.15, 1.6)

                # Style header
                for col_idx in range(len(headers)):
                    cell = table[(0, col_idx)]
                    cell.set_facecolor("#2b5b84")
                    cell.set_text_props(color="white", weight="bold")

                # Style Average row
                last_row_idx = len(table_data)
                for col_idx in range(len(headers)):
                    cell = table[(last_row_idx, col_idx)]
                    cell.set_facecolor("#e2eaf2")
                    cell.set_text_props(weight="bold")

                # Alternate row shading
                for row_idx in range(1, last_row_idx):
                    color = "#f9fbfd" if row_idx % 2 == 0 else "#ffffff"
                    for col_idx in range(len(headers)):
                        table[(row_idx, col_idx)].set_facecolor(color)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(f"[+] Clean PDF results generated at: {pdf_path}")

if __name__ == "__main__":
    create_pdf_results()
