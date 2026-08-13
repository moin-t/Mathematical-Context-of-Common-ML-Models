# Exploration Strategies in Tabular Q-Learning


This repository compares constant epsilon-greedy, decaying epsilon-greedy, softmax, and upper-confidence-bound (UCB) exploration for tabular Q-learning in a stochastic grid-world. Exact value iteration computes $Q^*$; ten independent seeds measure reward, success, and action-value error. 

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

Use Python 3.10.

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
report.pdf           Pdf technical report
report.md             GitHub-rendering technical report
run_experiments.py    complete experiment runner
```


