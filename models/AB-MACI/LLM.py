class LLM_Agent:
    def __init__(self, model):
        self.model = model  # LLM API or local model

    def predict_proba(self, X):
        """
        Returns probability distribution over {0,1,2,3,4} for each input sample.
        """
        P = []
        for x in X:
            response = self.model.query(f"Classify this input: {x}")  # LLM Call
            prob = parse_llm_response(response)  # Convert response to probability
            P.append(prob)
        return np.array(P)
