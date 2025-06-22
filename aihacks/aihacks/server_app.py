import flwr as fl
import numpy as np
import json
import joblib
from flwr.common import parameters_to_ndarrays
from task import make_model, FEATURES_PATH


class SaveModel(fl.server.strategy.FedAvg):
    def __init__(self, rounds=10):
        super().__init__(
            fraction_fit=0.85,
            min_fit_clients=50,
            min_available_clients=50
        )
        self.rounds = rounds
        self.params = None

    def aggregate_fit(self, rnd, results, failures):
        agg = super().aggregate_fit(rnd, results, failures)
        if agg:
            self.params = agg[0]

            if rnd == self.rounds:
                # Load feature names
                features = json.load(open(FEATURES_PATH))

                # Make model and initialize it with dummy fit
                model = make_model(len(features))
                dummy_X = np.zeros((2, len(features)))
                dummy_y = np.array([0, 1])
                model.partial_fit(dummy_X, dummy_y, classes=np.array([0, 1]))

                # Convert FL parameters → model weights
                weights = parameters_to_ndarrays(self.params)
                n_layers = len(model.coefs_)
                model.coefs_ = weights[:n_layers]
                model.intercepts_ = weights[n_layers:]

                # Save final model
                joblib.dump({"model": model, "features": features}, "fed_turnout.joblib")
                print("✅ saved fed_turnout.joblib")

        return agg


# Instantiate strategy
strategy = SaveModel(rounds=10)

# Start server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
