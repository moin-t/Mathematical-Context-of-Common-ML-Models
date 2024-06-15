import unittest
import numpy as np

from rl_exploration import GridWorld, value_iteration, train_q_learning


class ProjectTests(unittest.TestCase):
    def test_transition_probabilities_sum_to_one(self):
        env = GridWorld()
        for state in range(env.n_states):
            for action in range(env.n_actions):
                total = sum(x[0] for x in env.transitions(state, action))
                self.assertAlmostEqual(total, 1.0)

    def test_value_iteration_satisfies_bellman_equation(self):
        env = GridWorld()
        values, q_star = value_iteration(env)
        nonterminal = [s for s in range(env.n_states) if s not in env.terminal_states]
        np.testing.assert_allclose(values[nonterminal], q_star[nonterminal].max(axis=1), atol=1e-9)

    def test_q_learning_shapes_and_history(self):
        env = GridWorld()
        _, q_star = value_iteration(env)
        q_values, history = train_q_learning(env, "decaying_epsilon", q_star, episodes=20, eval_every=5, seed=7)
        self.assertEqual(q_values.shape, (env.n_states, env.n_actions))
        self.assertGreaterEqual(len(history), 4)
        self.assertTrue(np.isfinite(q_values).all())

    def test_training_is_reproducible(self):
        env = GridWorld()
        _, q_star = value_iteration(env)
        q1, _ = train_q_learning(env, "softmax", q_star, episodes=30, seed=3)
        q2, _ = train_q_learning(env, "softmax", q_star, episodes=30, seed=3)
        np.testing.assert_array_equal(q1, q2)


if __name__ == "__main__":
    unittest.main()
