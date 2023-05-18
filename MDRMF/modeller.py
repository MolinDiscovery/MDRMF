import numpy as np
from .dataset import Dataset

class Modeller:

    def __init__(self, dataset, iterations=10, initial_sample_size=10, acquisition_size=10, acquisition_method="greedy") -> None:
        
        self.dataset = dataset
        self.iterations = iterations
        self.initial_sample_size = initial_sample_size
        self.acquisition_size = acquisition_size
        self.acquisition_method = acquisition_method

    def initial_sampler(self):
        # Select random points in the dataset
        random_indices = np.random.choice(len(self.dataset.X), size=self.initial_sample_size, replace=False)

        # Select random points
        X_samples = self.dataset.X[random_indices]
        y_samples = self.dataset.y[random_indices]
        ids_samples = self.dataset.ids[random_indices]
        w_samples = self.dataset.w[random_indices]

        random_points = Dataset(X=X_samples, y=y_samples, ids=ids_samples, w=w_samples)

        # Delete selected points from dataset
        mask = np.ones(len(self.dataset.X), dtype=bool)
        mask[random_indices] = False
        self.dataset.X = self.dataset.X[mask]
        self.dataset.y = self.dataset.y[mask]
        self.dataset.ids = self.dataset.ids[mask]
        self.dataset.w = self.dataset.w[mask]

        return random_points

    def acquisition():
        pass
    
    def fit():
        pass

    def predict():
        pass