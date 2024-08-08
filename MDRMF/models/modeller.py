from cProfile import label
import numpy as np
import sys
from scipy.stats import norm
from MDRMF import model
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
    def __init__(
            self,
            dataset, 
            evaluator=None, 
            iterations=10, 
            initial_sample_size=10, 
            acquisition_size=10, 
            acquisition_method="greedy", 
            retrain=True,
            seeds=[],
            add_noise=None,
            model_graphs=False) -> None:
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
        self.add_noise = add_noise
        self.model_graphs = model_graphs

        self.results = {}
        self.figures = []
        self.model_datasets = []


    def _initial_sampler(self, initial_sample_size):
        """
        Randomly samples the initial points from the dataset.

        Returns:
            numpy.ndarray: Array of randomly selected points.
        """
        random_points = self.dataset.get_samples(initial_sample_size, remove_points=True)

        return random_points


    def _acquisition(self, model, model_dataset, add_noise: int = 0.1):
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

        if self.acquisition_method == "tanimoto_hits":

            hit_feature_vectors = model_dataset.X
            pred_feature_vectors = self.dataset.X

            arr = np.zeros((len(hit_feature_vectors), len(pred_feature_vectors)))

            for hit_index, hit_mol in enumerate(hit_feature_vectors):
                
                for pred_index, pred_mol in enumerate (pred_feature_vectors):

                    fp_hits = np.where(hit_mol == 1)[0]
                    fp_preds = np.where(pred_mol == 1)[0]

                    common = set(fp_hits) & set(fp_preds)
                    combined = set(fp_hits) | set(fp_preds)

                    similarity = len(common) / len(combined)
                    
                    arr[hit_index, pred_index] = similarity
            
            picks_idx = np.argsort(np.max(arr, axis=0))[::-1][:self.acquisition_size]
            
            acq_dataset = self.dataset.get_points(list(picks_idx))

        if self.acquisition_method == 'tanimoto':
            
            pred_feature_vectors = self.dataset.X

            model_dataset.sort_by_y()
            best_mol = model_dataset.X[0]

            arr = np.zeros(len(pred_feature_vectors))

            for pred_i, pred_mol in enumerate(pred_feature_vectors):
                
                fp_best = np.where(best_mol == 1)[0]
                fp_preds = np.where(pred_mol == 1)[0]

                common = set(fp_best) & set(fp_preds)
                combined = set(fp_best) | set(fp_preds)

                similarity = len(common) / len(combined)

                arr[pred_i] = similarity

            picks_idx = np.argsort(arr)[::-1][:self.acquisition_size]

            acq_dataset = self.dataset.get_points(list(picks_idx), remove_points=True)

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

        if self.acquisition_method == "EI":
            # Find indices of the x-number of smallest values
            low_pred_indices = np.argpartition(preds, self.acquisition_size)[:self.acquisition_size]

            # Calculate EI based on these selected predictions and their uncertainties
            selected_preds = preds[low_pred_indices]
            selected_uncertainty = uncertainty[low_pred_indices]
            best_so_far = np.min(model_dataset.y)  # Assuming this represents the best observed value so far
            ei_scores = self._calculate_ei(selected_preds, selected_uncertainty, best_so_far)

            # Prioritize samples with higher EI scores
            # Note: Since EI scores correspond to the already filtered low_pred_indices, we sort ei_scores and use them to reorder low_pred_indices
            ei_sorted_indices = np.argsort(-ei_scores)  # Higher EI scores first
            final_indices = low_pred_indices[ei_sorted_indices][:self.acquisition_size]

            acq_dataset = self.dataset.get_points(final_indices, remove_points=True)

        if self.acquisition_method == "TS":
            # TS stands for Thompson Sampling.

            # Sample from the predictive distribution
            samples = np.random.normal(preds, uncertainty)
            
            # Find the indices with the lowest sampled values
            indices = np.argpartition(samples, self.acquisition_size)[:self.acquisition_size]

            acq_dataset = self.dataset.get_points(indices, remove_points=True)


        # Below we add noise to the acquired dataset to simulate real world lab data.
        if add_noise is not None:
            noises = np.random.normal(0, add_noise, size=acq_dataset.y.size)
            acq_dataset.y = acq_dataset.y + noises

        return acq_dataset
    
    
    def unlabeled_acquisition(self, model, dataset, dataset_labeled):
        """
        Performs the acquisition step to select new points for testing.

        Parameters:
            model: The model object used for acquisition.

        Returns:
            Dataset: The acquired dataset containing the selected points.
        """
        # Predict on the full dataset
        preds = self.predict(dataset)
        

        if self.acquisition_method == "greedy":

            # Find indices of the x-number of smallest values
            indices = np.argpartition(-preds, self.acquisition_size)[:self.acquisition_size]

            # Get the best docked molecules from the dataset
            acq_dataset = dataset.get_points(indices, remove_points=False, unlabeled=True)

        if self.acquisition_method == "random":
            
            # Get random points
            acq_dataset = dataset.get_samples(self.acquisition_size, remove_points=False, unlabeled=True)

        if self.acquisition_method == 'tanimoto':
            
            pred_feature_vectors = dataset.X

            dataset_labeled.sort_by_y(ascending=False)
            best_mol = dataset_labeled.X[0]

            arr = np.zeros(len(pred_feature_vectors))

            for pred_i, pred_mol in enumerate(pred_feature_vectors):
                
                fp_best = np.where(best_mol == 1)[0]
                fp_preds = np.where(pred_mol == 1)[0]

                common = set(fp_best) & set(fp_preds)
                combined = set(fp_best) | set(fp_preds)

                similarity = len(common) / len(combined)

                arr[pred_i] = similarity

            picks_idx = np.argsort(arr)[::-1][:self.acquisition_size]

            acq_dataset = dataset.get_points(list(picks_idx), remove_points=False, unlabeled=True)            

        return acq_dataset


    def _calculate_ei(self, selected_preds, selected_uncertainty, best_so_far):
        """
        Calculate the Expected Improvement (EI) for a subset of selected samples based on their predictions and uncertainties.

        Parameters:
            selected_preds (numpy.ndarray): The model's predictions for the selected samples.
            selected_uncertainty (numpy.ndarray): The model's prediction uncertainties for the selected samples.
            best_so_far (float): The best (lowest) prediction value observed across all samples.

        Returns:
            numpy.ndarray: The EI for each selected sample.
        """
        # Ensure no division by zero
        mask = selected_uncertainty > 0
        improvement = np.zeros(selected_preds.shape)
        improvement[mask] = best_so_far - selected_preds[mask]

        # Safe division
        z = np.zeros(selected_preds.shape)
        z[mask] = improvement[mask] / selected_uncertainty[mask]

        # Calculate EI
        ei = np.zeros(selected_preds.shape)
        ei[mask] = improvement[mask] * norm.cdf(z[mask]) + selected_uncertainty[mask] * norm.pdf(z[mask])

        return ei


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
        self.model_datasets.append(model_dataset) # appends the model_dataset so it can be exported to the results folder.


    def graph_model(self):
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D  # Import for custom legend handles

        dataset = self.dataset
        model_dataset = self.model_dataset
        
        preds = self.predict(dataset)
        preds_model = self.predict(model_dataset)
        
        fig, ax = plt.subplots(dpi=300)

        # Plot the truth line and keep a reference to it for the legend, choose a preferred color
        truth_line = ax.plot(dataset.y, dataset.y, label='Truth line', color='tab:red')  # Updated color
        # Plot the scatter plots with small dots, choose colors you like
        scatter1 = ax.scatter(dataset.y, preds, label='Unlabelled predictions', s=1, color='tab:purple')  # Updated color
        scatter2 = ax.scatter(model_dataset.y, preds_model, label='Labelled Predictions', s=1, color='tab:cyan')  # Updated color

        ax.set_xlabel('Truth')
        ax.set_ylabel('Predictions')

        # Create custom legend handles, adjust the truth line to not use a marker and use a solid line
        legend_handles = [
            Line2D([0], [0], color='tab:red', lw=2, label='Truth line'),  # Solid line for truth line
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:purple', markersize=10, label='Unlabelled predictions'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:cyan', markersize=10, label='Labelled Predictions'),
        ]

        # Add the custom legend handles to the legend
        ax.legend(handles=legend_handles)

        self.figures.append(fig)

