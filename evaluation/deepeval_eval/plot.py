# pylint: disable=C0114
# pylint: disable=C0116

import os
import sys
import json
from typing import (
    List, Dict, Final
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

METRICS: Final[List[str]] = [
    "Answer Relevancy",
    "Faithfulness",
    "Contextual Precision",
    "Contextual Recall",
    "Contextual Relevancy"
]

CUSTOM_LABELS = {
    "Answer Relevancy": "Answer Relevance",
    "Faithfulness": "Faithfulness",
    "Contextual Precision": "Context Precision",
    "Contextual Recall": "Context Recall",
    "Contextual Relevancy": "Context Relevance"
}

COLOR_MAP = {
    "Answer Relevancy": "#2ca02c",
    "Faithfulness": "#d62728",
    "Contextual Precision": "#1f77b4",
    "Contextual Recall": "#ff7f0e",
    "Contextual Relevancy": "#9467bd"
}

def validate_user_input() -> str:
    try:
        path: str = sys.argv[1]
        return path
    except IndexError:
        print(f"Usage: python {sys.argv[0]} <dirpath_to_experiment_results>")
        sys.exit(1)

def fetch_sorted_filenames(dirpath: str) -> List[str]:
    files: List[str] = os.listdir(dirpath)
    # Sort files numerically based on the leading number in the filename
    files.sort(key=lambda x: int(x.split('_')[0]))
    return files

def load_file_content(filename: str) -> Dict[str, List[Dict]]:
    metrics_data: Dict[str, List[Dict]] = {}

    with open(file=filename, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            data = json.loads(line)
            metric_name = data["metric"]
            metric_scores = data["scores"]
            metrics_data[metric_name] = metric_scores

    return metrics_data

def compute_mean_score(samples: List[Dict]) -> float:
    scores: List[float] = [s["score"] for s in samples if s["score"] is not None]
    return sum(scores) / len(scores) if scores else 0.0

def plot_experiment_scores(experiment_scores: Dict[str, Dict[str, float]]) -> None:
    if not experiment_scores:
        raise ValueError("No experiment scores found!")

    experiments = sorted(experiment_scores.keys(), key=int)
    num_experiments = len(experiments)
    num_metrics = len(METRICS)

    width = 0.15
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
            color=COLOR_MAP.get(metric, colors[idx]),
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
        label="RAG System Performance Metrics Across Experiments (DeepEval)",
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(len(METRICS), 5),
        fontsize=10,
        frameon=False
    )

    plt.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
    plt.grid(axis="x", linestyle=":", alpha=0.2)

    plt.ylim(0, 1.08)
    plt.gca().set_facecolor('#FCFCFC')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1)

    output_path = "../../img/plots/deepeval_summary_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

def main():
    path: str = validate_user_input()

    files: List[str] = fetch_sorted_filenames(path)

    # Maps the experiment id to its metrics (metric name -> average score)
    experiment_scores: Dict[str, Dict[str, float]] = {}

    # # Iterate over the files
    for file in files:
        filename: str = os.path.join(path, file)
        results: dict = load_file_content(filename)
        print(f"PROCESSING: {file}")

        experiment_id: str = file.split("_")[0]
        experiment_scores[experiment_id] = {}

        # Iterate over all the expected metrics to take the average per file
        for metric in METRICS:
            if metric not in results:
                raise ValueError(f"Metric {metric} not found in file {filename}!")

            samples: List[Dict] = results[metric]
            mean: float = compute_mean_score(samples)
            experiment_scores[experiment_id][metric] = mean

    plot_experiment_scores(experiment_scores)

if __name__ == "__main__":
    main()
