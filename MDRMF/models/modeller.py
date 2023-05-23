import numpy as np
from ..dataset import Dataset

class Modeller:
    """
    Base class to construct other models from
    """
    def __init__(self, dataset, iterations=10, initial_sample_size=10, acquisition_size=10, acquisition_method="greedy", retrain=True) -> None:
        
        self.dataset = dataset
        self.iterations = iterations
        self.initial_sample_size = initial_sample_size
        self.acquisition_size = acquisition_size
        self.acquisition_method = acquisition_method
        self.retrain = retrain

    def _initial_sampler(self):
        # Select random points in the dataset
        random_indices = np.random.choice(len(self.dataset.X), size=self.initial_sample_size, replace=False)

        # Select random points
        random_points = self.dataset.get_points(random_indices)

        # Delete selected points from dataset
        self.dataset.remove_points(random_indices)


        return random_points

    def _acquisition(self, model):
        if self.acquisition_method == "greedy":
            
            # Predict on the full dataset
            preds = model.predict(self.dataset.X)

            # Find indices of the x-number of smallest values
            indices = np.argpartition(preds, self.acquisition_size)[:self.acquisition_size]

            # Get the best docked molecules from the dataset
            acq_dataset = self.dataset.get_points(indices)

            # Remove these datapoints from the dataset
            self.dataset.remove_points(indices)

            return acq_dataset
    
    def fit():
        pass # Must be defined in child classes

    def predict():
        pass # Must be defined in child classes