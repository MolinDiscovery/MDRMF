import numpy as np
from ..dataset import Dataset

class Modeller:
    """
    Base class to construct other models from
    """
    def __init__(self, dataset, evaluator=None, iterations=10, initial_sample_size=10, acquisition_size=10, acquisition_method="greedy", retrain=True) -> None:
        
        self.dataset = dataset
        self.evaluator = evaluator
        self.iterations = iterations
        self.initial_sample_size = initial_sample_size
        self.acquisition_size = acquisition_size
        self.acquisition_method = acquisition_method
        self.retrain = retrain
        self.results = {}

    def _initial_sampler(self):

        random_points = self.dataset.get_samples(self.initial_sample_size, remove_points=True)

        return random_points

    def _acquisition(self, model):
        # Predict on the full dataset
        preds = model.predict(self.dataset.X)

        if self.acquisition_method == "greedy":

            # Find indices of the x-number of smallest values
            indices = np.argpartition(preds, self.acquisition_size)[:self.acquisition_size]

            # Get the best docked molecules from the dataset
            acq_dataset = self.dataset.get_points(indices)

            # Remove these datapoints from the dataset
            self.dataset.remove_points(indices)

        if self.acquisition_method == "random":
            
            # Get random points and delete from dataset
            acq_dataset = self.dataset.get_samples(self.acquisition_size, remove_points=True)

        return acq_dataset
    
    def fit(self):
        pass # Must be defined in child classes

    def predict():
        pass # Must be defined in child classes
    
    def call_evaluator(self, i):
        results = self.evaluator.evaluate(self, self.dataset)
        print(f"Iteration {i}, Results: {results}")

        # Store results
        self.results[i] = results