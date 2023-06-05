import numpy as np
from MDRMF.dataset import Dataset

class Modeller:
    """
    Base class to construct other models from
    
    Parameters:
        dataset (Dataset): The dataset object containing the data.
        evaluator (Evaluator): The evaluator object used to evaluate the model's performance.
        iterations (int): The number of iterations to perform.
        initial_sample_size (int): The number of initial samples to randomly select from the dataset.
        acquisition_size (int): The number of points to acquire in each iteration.
        acquisition_method (str): The acquisition method to use, either "greedy" or "random".
        retrain (bool): Flag indicating whether to retrain the model in each iteration.
    """
    def __init__(self, dataset, evaluator=None, iterations=10, initial_sample_size=10, acquisition_size=10, acquisition_method="greedy", retrain=True) -> None:
        """
        Initializes a Modeller object with the provided parameters.
        """        
        self.dataset = dataset.copy()
        self.evaluator = evaluator
        self.iterations = iterations
        self.initial_sample_size = initial_sample_size
        self.acquisition_size = acquisition_size
        self.acquisition_method = acquisition_method
        self.retrain = retrain
        self.results = {}

    def _initial_sampler(self):
        """
        Randomly samples the initial points from the dataset.

        Returns:
            numpy.ndarray: Array of randomly selected points.
        """
        random_points = self.dataset.get_samples(self.initial_sample_size, remove_points=True)

        return random_points

    def _acquisition(self, model):
        """
        Performs the acquisition step to select new points for the model.

        Parameters:
            model: The model object used for acquisition.

        Returns:
            Dataset: The acquired dataset containing the selected points.
        """

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
        """
        Fits the model to the data.
        This method needs to be implemented in child classes.
        """        
        pass

    def predict():
        """
        Generates predictions using the fitted model.
        This method needs to be implemented in child classes.
        """        
        pass
    
    def call_evaluator(self, i):
        """
        Calls the evaluator to evaluate the model's performance and stores the results.

        Parameters:
            i (int): The current iteration number.

        
        Notes: Should always be called when defining the fit() in a child model.
        """
        results = self.evaluator.evaluate(self, self.dataset)
        print(f"Iteration {i+1}, Results: {results}")

        # Store results
        self.results[i+1] = results