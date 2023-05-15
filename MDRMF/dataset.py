# dataset.py

import numpy as np
import pickle

class Dataset:

    def __init__(self, X, y, ids=None, w=None) -> None:
        
        n_samples = np.shape(X)[0]

        if w is None:
            if len(y.shape) == 1:
                w = np.ones(y.shape[0], np.float32)
            else:
                w = np.ones((y.shape[0], 1), np.float32)

        if ids is None:
            ids = np.arange(n_samples)

        self.X = X
        self.y = y
        self.ids = np.array(ids, dtype=object)
        self.w = w

    def __repr__(self):
        return f"<Dataset X.shape: {self.X.shape}, y.shape: {self.y.shape}, w.shape: {self.w.shape}, ids: {self.ids}>"
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)