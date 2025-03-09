import numpy as np

class CustomAdaBoostLLM:
    def __init__(self, agents, num_iterations=10):
        self.agents = agents  # List of LLM classifiers
        self.num_iterations = num_iterations
        self.alpha = []

    def fit(self, X, y):
        N = len(X)
        w = np.ones(N) / N  # Initialize weights

        for t, agent in enumerate(self.agents):
            P_t = agent.predict_proba(X)  # Get LLM output probabilities (N x 5)

            e_t = np.sum(w * (1 - P_t[np.arange(N), y]))  # Weighted error
            alpha_t = 0.5 * np.log((1 - e_t) / e_t)
            self.alpha.append(alpha_t)

            # Update sample weights
            w *= np.exp(alpha_t * (1 - P_t[np.arange(N), y]))
            w /= np.sum(w)  # Normalize

    def predict(self, X):
        final_prob = np.zeros((len(X), 5))

        for t, agent in enumerate(self.agents):
            P_t = agent.predict_proba(X)  # Get probability distribution
            final_prob += self.alpha[t] * P_t  # Weight by alpha

        return np.argmax(final_prob, axis=1)  # Final class prediction
