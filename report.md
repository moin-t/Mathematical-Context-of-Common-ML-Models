# Exploration Strategies in Tabular Q-Learning

## Technical Report

**Author:** Your Name  
**Programme:** BS Mathematics  
**Repository:** `https://github.com/your-username/rl-exploration-project`

## Abstract

This project studies how exploration affects tabular Q-learning in a finite stochastic grid-world. Four strategies are compared: constant epsilon-greedy, decaying epsilon-greedy, softmax exploration, and the upper-confidence-bound rule. Exact value iteration supplies the optimal action-value function used to measure estimation error. Each learning method is evaluated over ten random seeds using greedy-policy return, success rate, and maximum action-value error. The experiment is intentionally small and transparent, making it reproducible on an ordinary computer and suitable for an undergraduate study of Markov decision processes, contraction mappings, stochastic approximation, and statistical comparison.

## 1. Introduction

Reinforcement learning concerns an agent that repeatedly observes a state, chooses an action, and receives a reward. The agent must balance exploitation of actions currently believed to be good against exploration of less-known actions. Too little exploration can trap learning at a poor policy; too much exploration can waste samples and lower reward.

The research question is: **How does the exploration rule affect the convergence, stability, and performance of tabular Q-learning in a finite Markov decision process?**

## 2. Mathematical framework

A discounted finite Markov decision process is a tuple `(S, A, P, R, gamma)`. For state `s` and action `a`, `P(s'|s,a)` is the probability of the next state, `R(s,a,s')` is the immediate reward, and `0 <= gamma < 1` discounts later rewards. The optimal action-value function obeys

`Q*(s,a) = sum_s' P(s'|s,a)[R(s,a,s') + gamma max_b Q*(s',b)].`

The associated Bellman optimality operator is a contraction in the maximum norm:

`||TQ - TU||_infinity <= gamma ||Q - U||_infinity.`

Because the space of finite Q-tables is complete and `gamma < 1`, the Banach fixed-point theorem gives a unique fixed point `Q*`. Repeated exact Bellman updates converge to this fixed point.

Q-learning replaces the expectation with sampled transitions:

`Q_(t+1)(s_t,a_t) = Q_t(s_t,a_t) + alpha[R_(t+1) + gamma max_a Q_t(s_(t+1),a) - Q_t(s_t,a_t)].`

Classical convergence requires every state-action pair to be visited infinitely often and learning rates satisfying `sum alpha_t = infinity` and `sum alpha_t^2 < infinity`. This experiment uses a fixed learning rate for practical finite-sample comparison; consequently, exact asymptotic convergence is not claimed.

## 3. Exploration strategies

1. **Constant epsilon-greedy:** take a uniformly random action with probability 0.10 and a greedy action otherwise.
2. **Decaying epsilon-greedy:** begin at epsilon 1.0, multiply it by 0.995 per episode, and stop at 0.02.
3. **Softmax:** sample actions in proportion to `exp(Q(s,a)/tau)`, with temperature decaying from 1.0 to 0.05.
4. **UCB:** select the action maximizing `Q(s,a) + c sqrt(log N(s)/N(s,a))`, with `c=1`. Untried actions receive priority.

## 4. Experimental design

The environment is a 6-by-6 grid. The agent begins at the lower-left corner and seeks a goal at the upper-right. Five hazard cells terminate an episode with reward -10; the goal gives +10; every ordinary step gives -0.1. The requested action occurs with probability 0.90, while each other direction occurs with probability 0.10/3. Episodes are truncated after 100 steps.

Each method trains for 1,500 episodes with discount factor 0.95 and learning rate 0.15. Every 25 episodes, its greedy policy is evaluated for 40 episodes. Ten independent training seeds are used. Shaded chart regions are approximate 95% confidence intervals for the mean across seeds.

The outcome measures are:

- greedy-policy mean return;
- probability of reaching the goal;
- maximum absolute Q-error over nonterminal state-action pairs;
- variability across seeds.

## 5. Results

Run `python run_experiments.py` to reproduce the numerical results. The generated summary is in `results/summary.csv`; complete observations are in `results/learning_curves.csv`. Figures show the learning trajectory for return, success rate, and Q-error. Differences should be interpreted as evidence for this environment and hyperparameter setting, not universal rankings.

## 6. Discussion

Constant epsilon-greedy continues to explore indefinitely, which protects against premature commitment but can slow accurate value estimation. Decaying epsilon shifts gradually from exploration to exploitation. Softmax uses relative value estimates rather than treating all exploratory actions equally, although it is sensitive to temperature. UCB directs exploration toward uncertain actions through visit counts, but the standard bandit bonus does not account fully for delayed state transitions.

The optimal-value error can remain appreciable even when the greedy policy succeeds reliably. This is not contradictory: several inaccurate action values may leave the identity of the best action unchanged. Policy performance and value estimation therefore answer different questions and should both be reported.

## 7. Limitations and extensions

The study uses one small MDP, one hyperparameter setting per method, a constant learning rate, and a simple confidence interval across only ten seeds. Hyperparameters were not tuned on a separate validation task. A stronger follow-up could use Robbins-Monro learning rates, compare additional environments, report cumulative regret, use paired statistical tests, or study offline learning from fixed datasets.

## 8. Conclusion

This project demonstrates that exploration is a measurable design choice rather than a minor implementation detail. It also connects an exact contraction-based solution of a finite MDP with a sample-based stochastic learning algorithm. The combination of mathematical benchmark, transparent implementation, repeated trials, and cautious interpretation makes the study a suitable undergraduate portfolio project.

## References

1. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, second edition. MIT Press.
2. Watkins, C. J. C. H., and Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279-292.
3. Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47, 235-256.
4. Farama Foundation. *Gymnasium documentation*. https://gymnasium.farama.org/
