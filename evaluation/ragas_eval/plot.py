# pylint: disable=C0114
# pylint: disable=C0116

import os
import sys
from typing import (
    List, Dict, Final
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

METRICS: Final[List[str]] = [
    "context_precision",
    "context_recall",
    "context_entity_recall",
    "noise_sensitivity(mode=relevant)",
    "answer_relevancy",
    "faithfulness",
    "factual_correctness(mode=f1)"
]

CUSTOM_LABELS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "context_entity_recall": "Context Entity Recall",
    "noise_sensitivity(mode=relevant)": "Noise Sensitivity (relevant)",
    "answer_relevancy": "Answer Relevancy",
    "faithfulness": "Faithfulness",
    "factual_correctness(mode=f1)": "Factual Correctness"
}

def validate_user_input() -> str:
    try:
        path: str = sys.argv[1]
        return path
    except IndexError:
        print(f"Usage: python {sys.argv[0]} <dirpath_to_experiment_results>")
        sys.exit(1)

def fetch_sorted_filenames(path: str) -> List[str]:
    files: List[str] = os.listdir(path)
    # Sort files numerically based on the leading number in the filename
    files.sort(key=lambda x: int(x.split('_')[0]))
    return files

def load_file_content(filename: str) -> Dict[str, List[Dict]]:
    df: pd.DataFrame = pd.read_csv(filename)
    return df[METRICS].to_dict(orient="list")

def compute_mean_score_per_metric(scores: List[float]) -> float:
    return float(np.nanmean(scores))

def plot_experiment_scores(experiment_scores: Dict[str, Dict[str, float]]) -> None:
    if not experiment_scores:
        raise ValueError("No experiment scores found!")

    experiments = sorted(experiment_scores.keys(), key=int)
    num_experiments = len(experiments)
    num_metrics = len(METRICS)

    width = 0.12
    x = np.arange(num_experiments)

    plt.figure(figsize=(24, 8))

    cmap = get_cmap("tab20")
    colors = [cmap(i) for i in range(len(METRICS))]

    for idx, metric in enumerate(METRICS):
        scores = [experiment_scores[exp_id].get(metric, 0.0) for exp_id in experiments]
        offset_x = x + (idx - num_metrics / 2) * width + width / 2

        bars = plt.bar(
            offset_x,
            scores,
            width=width,
            color=colors[idx],
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5,
            label=CUSTOM_LABELS.get(metric, metric.replace("_", " ").title())
        )

        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                offset_per_metric = min(0.015 + idx * 0.015, 0.07)
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + offset_per_metric,
                    f"{height:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='medium'
                )

    plt.xticks(x, [f"Experiment {exp}" for exp in experiments], fontsize=11)
    plt.ylabel("Average Score", fontsize=13, fontweight='medium')
    plt.xlabel("Experiment ID", fontsize=13, fontweight='medium')
    plt.title(
        label="RAG System Performance Metrics Across Experiments (RAGAs)",
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(len(METRICS), 10),
        fontsize=10,
        frameon=False
    )

    plt.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    plt.grid(axis="x", linestyle=":", alpha=0.2)

    plt.ylim(0, 0.95)
    plt.gca().set_facecolor('#FCFCFC')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1)

    plt.savefig("../../img/plots/ragas_summary_plot.png", dpi=300, bbox_inches="tight")
    print("Plot saved to ../../img/plots/ragas_summary_plot.png")

def main():
    dir_path: str = validate_user_input()

    filenames: List[str] = fetch_sorted_filenames(dir_path)

    # Maps the experiment id to its metrics (metric name -> average score)
    experiment_scores: Dict[str, Dict[str, float]] = {}

    # Iterate over the files
    for filename in filenames:
        full_fillename: str = os.path.join(dir_path, filename)
        results: dict = load_file_content(full_fillename)
        print(f"PROCESSING: {full_fillename}")

        experiment_id: str = filename.split("_")[0]
        experiment_scores[experiment_id] = {}

        # Iterate over all the expected metrics to take the average per file
        for metric in METRICS:
            if metric not in results:
                raise ValueError(f"Metric {metric} not found in file {filename}!")

            samples: List[float] = results[metric]
            mean: float = compute_mean_score_per_metric(samples)
            experiment_scores[experiment_id][metric] = mean

    plot_experiment_scores(experiment_scores)

if __name__ == "__main__":
    main()
