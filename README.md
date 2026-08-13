# Exploration Strategies in Tabular Q-Learning

> **Undergraduate mathematics project — 2023 edition**

This repository compares constant epsilon-greedy, decaying epsilon-greedy, softmax, and upper-confidence-bound (UCB) exploration for tabular Q-learning in a stochastic grid-world. Exact value iteration computes $Q^*$; ten independent seeds measure reward, success, and action-value error. The implementation avoids RL frameworks so every mathematical step remains visible.

## Research question

**How does exploration affect the convergence, stability, and performance of tabular Q-learning in a finite stochastic Markov decision process?**

## Mathematics

The optimal action-value function satisfies

$$
Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left[R(s,a,s')+\gamma\max_b Q^*(s',b)\right].
$$

Q-learning uses the sampled update

$$
Q_{t+1}(S_t,A_t)=Q_t(S_t,A_t)+\alpha\left[R_{t+1}+\gamma\max_aQ_t(S_{t+1},a)-Q_t(S_t,A_t)\right].
$$

See [report.md](report.md) for the complete derivation, experiment, results, limitations, and references.

## Results

| Strategy | Mean return | Success | Maximum Q-error |
|---|---:|---:|---:|
| Constant epsilon-greedy | 5.02 | 82.5% | **8.64** |
| Decaying epsilon-greedy | 4.08 | 75.5% | 9.07 |
| Softmax | 6.41 | 87.0% | 9.30 |
| UCB | **7.17** | **91.0%** | 9.41 |

![Evaluation return](figures/evaluation_return.png)

## Install and reproduce

The pinned stack was available by 2023. Use Python 3.10.

```bash
git clone https://github.com/your-username/rl-exploration-project.git
cd rl-exploration-project
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_experiments.py
python -m unittest discover -s tests -v
```

## Repository structure

```text
rl_exploration/       environment, planning, policies, and Q-learning
tests/                correctness and reproducibility tests
figures/              generated learning curves
results/              raw and summary CSV data
report.docx           Microsoft Word technical report
report.md             GitHub-rendering technical report
run_experiments.py    complete experiment runner
```

## 2023 compatibility

The project uses Python 3.10, NumPy 1.23.5, Matplotlib 3.6.3, and python-docx 0.8.11. All methods and references were available by the end of 2023. Gymnasium is not imported. See [HISTORICAL_COMPATIBILITY.md](HISTORICAL_COMPATIBILITY.md).

## Before publishing

1. Replace the author, institution, supervisor, and repository placeholders.
2. Rerun the experiment and retain the generated CSV files.
3. Be prepared to explain the Bellman contraction, Q-learning update, results, and limitations.
4. Create a GitHub repository and a `v1.0.0` release; optionally archive that release with Zenodo.

The source code is licensed under MIT. Update `CITATION.cff` before publication.
