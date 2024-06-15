"""Run the complete multi-seed study and save CSV data and figures."""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from rl_exploration import GridWorld, value_iteration, train_q_learning


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
STRATEGIES = ["epsilon_greedy", "decaying_epsilon", "softmax", "ucb"]
LABELS = {
    "epsilon_greedy": "Constant epsilon-greedy",
    "decaying_epsilon": "Decaying epsilon-greedy",
    "softmax": "Softmax",
    "ucb": "UCB",
}


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(episodes=1500, seeds=range(10)):
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    env = GridWorld()
    _, q_star = value_iteration(env)
    all_rows = []
    for strategy in STRATEGIES:
        for seed in seeds:
            _, history = train_q_learning(env, strategy, q_star, episodes=episodes, seed=seed)
            for row in history:
                all_rows.append({"strategy": strategy, "seed": seed, **row})
            print(f"completed {strategy}, seed {seed}")

    fields = ["strategy", "seed", "episode", "training_return", "evaluation_return", "success_rate", "q_error"]
    save_csv(RESULTS / "learning_curves.csv", all_rows, fields)

    final_rows = []
    for strategy in STRATEGIES:
        rows = [r for r in all_rows if r["strategy"] == strategy and r["episode"] == episodes]
        final_rows.append({
            "strategy": LABELS[strategy],
            "mean_evaluation_return": np.mean([r["evaluation_return"] for r in rows]),
            "std_evaluation_return": np.std([r["evaluation_return"] for r in rows], ddof=1),
            "mean_success_rate": np.mean([r["success_rate"] for r in rows]),
            "mean_q_error": np.mean([r["q_error"] for r in rows]),
        })
    save_csv(RESULTS / "summary.csv", final_rows, list(final_rows[0]))

    metrics = [
        ("evaluation_return", "Greedy evaluation return", "evaluation_return.png"),
        ("success_rate", "Greedy-policy success rate", "success_rate.png"),
        ("q_error", "Maximum Q-value error", "q_error.png"),
    ]
    for metric, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for strategy in STRATEGIES:
            subset = [r for r in all_rows if r["strategy"] == strategy]
            episode_values = sorted({r["episode"] for r in subset})
            matrix = np.array([[r[metric] for r in subset if r["seed"] == seed] for seed in seeds])
            mean, stderr = matrix.mean(axis=0), matrix.std(axis=0, ddof=1) / np.sqrt(len(seeds))
            ax.plot(episode_values, mean, label=LABELS[strategy], linewidth=2)
            ax.fill_between(episode_values, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.15)
        ax.set(xlabel="Training episode", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=180)
        plt.close(fig)
    print(f"Results saved in {RESULTS}")


if __name__ == "__main__":
    main()
