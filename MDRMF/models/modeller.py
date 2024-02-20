import numpy as np
import sys
from MDRMF.dataset import Dataset
from rdkit import DataStructs

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
    def __init__(
            self, 
            dataset, 
            evaluator=None, 
            iterations=10, 
            initial_sample_size=10, 
            acquisition_size=10, 
            acquisition_method="greedy", 
            retrain=True,
            seeds=[]) -> None:
        """
        Initializes a Modeller object with the provided parameters.
        """        
        self.dataset = dataset.copy()
        self.eval_dataset = dataset.copy()
        self.evaluator = evaluator
        self.iterations = iterations
        self.initial_sample_size = initial_sample_size
        self.acquisition_size = acquisition_size
        self.acquisition_method = acquisition_method
        self.retrain = retrain
        self.seeds = seeds
        self.results = {}


    def _initial_sampler(self, initial_sample_size):
        """
        Randomly samples the initial points from the dataset.

        Returns:
            numpy.ndarray: Array of randomly selected points.
        """
        random_points = self.dataset.get_samples(initial_sample_size, remove_points=True)

        return random_points


    def _acquisition(self, model, model_dataset):
        """
        Performs the acquisition step to select new points for the model.

        Parameters:
            model: The model object used for acquisition.

        Returns:
            Dataset: The acquired dataset containing the selected points.
        """

        # Predict on the full dataset
        # preds = model.predict(self.dataset.X)
        preds, uncertainty = self.predict(self.dataset, self.model_dataset, return_uncertainty=True)

        if self.acquisition_method == "greedy":

            # Find indices of the x-number of smallest values
            indices = np.argpartition(preds, self.acquisition_size)[:self.acquisition_size]

            # Get the best docked molecules from the dataset
            acq_dataset = self.dataset.get_points(indices, remove_points=True)

        if self.acquisition_method == "random":
            
            # Get random points and delete from dataset
            acq_dataset = self.dataset.get_samples(self.acquisition_size, remove_points=True)

        if self.acquisition_method == "tanimoto":

            hit_feature_vectors = model_dataset.X
            pred_feature_vectors = self.dataset.X

            arr = np.zeros((len(hit_feature_vectors), len(pred_feature_vectors)))

            for hit_index, hit_mol in enumerate(hit_feature_vectors):
                
                for pred_index, pred_mol in enumerate(pred_feature_vectors):

                    fp_hits = np.where(hit_mol == 1)[0]
                    fp_preds = np.where(pred_mol == 1)[0]

                    common = set(fp_hits) & set(fp_preds)
                    combined = set(fp_hits) | set(fp_preds)

                    similarity = len(common) / len (combined)
                    
                    arr[hit_index, pred_index] = similarity
            
            picks_idx = np.argsort(np.max(arr, axis=0))[::-1][:self.acquisition_size]
            
            acq_dataset = self.dataset.get_points(list(picks_idx))

        if self.acquisition_method == "MU":
            # MU stands for most uncertainty.

            # Finds the indices with the highest uncertainty.
            indices = np.argpartition(uncertainty, -self.acquisition_size)[-self.acquisition_size:]

            acq_dataset = self.dataset.get_points(indices, remove_points=True)

        if self.acquisition_method == 'LCB':
            # LCB stands for Lower Confidence Bound.
            
            # Calculate the LCB score for each point.
            beta = 1  # This is a hyperparameter that can be tuned.
            lcb = preds - beta * uncertainty  # Note: Assuming lower preds are better.
            
            # Find the indices with the lowest LCB score.
            # Since np.argpartition finds indices for the smallest values and we're minimizing, it's directly applicable here.
            indices = np.argpartition(lcb, self.acquisition_size)[:self.acquisition_size]
            
            acq_dataset = self.dataset.get_points(indices, remove_points=True)

        return acq_dataset
    
    
    def unlabeled_acquisition(self, model, dataset):
        """
        Performs the acquisition step to select new points for testing.

        Parameters:
            model: The model object used for acquisition.

        Returns:
            Dataset: The acquired dataset containing the selected points.
        """
        # Predict on the full dataset
        preds = model.predict(dataset)

        if self.acquisition_method == "greedy":

            # Find indices of the x-number of smallest values
            indices = np.argpartition(preds, self.acquisition_size)[:self.acquisition_size]

            # Get the best docked molecules from the dataset
            acq_dataset = dataset.get_points(indices, remove_points=False, unlabeled=True)

        if self.acquisition_method == "random":
            
            # Get random points
            acq_dataset = dataset.get_samples(self.acquisition_size, remove_points=False, unlabeled=True)

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


    def save():
        """
        Save the model
        This method needs to be implemented in child classes.
        """         
        pass


    def load():
        """
        Load the model
        This method needs to be implemented in child classes.
        """ 
        pass
    

    def call_evaluator(self, i, model_dataset):
        """
        Calls the evaluator to evaluate the model's performance and stores the results.

        Parameters:
            i (int): The current iteration number.

        
        Notes: Should always be called when defining the fit() in a child model.
        """
        results = self.evaluator.evaluate(self, self.eval_dataset, model_dataset)
        print(f"Iteration {i+1}, Results: {results}")

        # Store results
        self.results[i+1] = results