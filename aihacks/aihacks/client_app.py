import flwr as fl
import glob
import os
from task import build_xy, make_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.utils import shuffle
import numpy as np

class SklearnClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # 💡 Force model initialization
        dummy_X, dummy_y = shuffle(X_train[:10], y_train[:10], random_state=0)
        self.model.partial_fit(dummy_X, dummy_y, classes=np.unique(y_train))


    def get_parameters(self, config):
        # Return *all* weight matrices + biases as flat NumPy arrays
        coefs = [w.astype(np.float32) for w in self.model.coefs_]
        intercepts = [b.astype(np.float32) for b in self.model.intercepts_]
        return coefs + intercepts

    def set_parameters(self, parameters):
        n_layers = len(self.model.coefs_)
        self.model.coefs_ = parameters[:n_layers]
        self.model.intercepts_ = parameters[n_layers:]


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.partial_fit(self.X_train, self.y_train, classes=[0, 1])
        return self.get_parameters(config), len(self.X_train), {}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        return float(1 - acc), len(self.X_test), {"accuracy": float(acc)}


def start_client(csv_path):
    X, y = build_xy(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    model = make_model(X.shape[1])
    client = SklearnClient(model, X_train, X_test, y_train, y_test)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

import multiprocessing

if __name__ == "__main__":
    csv_files = glob.glob("result/*.csv")

    processes = []
    for file_path in csv_files:
        p = multiprocessing.Process(target=start_client, args=(file_path,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

