from cProfile import label
import numpy as np
from numpy import newaxis, concatenate
import sys
import logging
import pickle
import os
from scipy.stats import norm
from typing import Dict
from MDRMF.dataset import Dataset
from MDRMF.models.engine import Engine
from joblib import Parallel, delayed, parallel_config

num_cores = os.cpu_count()

class Modeller:

    def __init__(
            self,
            dataset,
            engine="RF",
            evaluator=None, 
            iterations=10, 
            initial_sample_size=10, 
            acquisition_size=10, 
            acquisition_method="greedy", 
            retrain=True,
            seeds=[],
            add_noise=None,
            model_graphs=False,
            feature_importance_opt=None,
            use_pairwise=False,
            **kwargs) -> None:
        
        self.engine_name = engine # used for a retraining later
        self.engine = Engine(self.engine_name, **kwargs)
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
        self.feature_importance_opt = feature_importance_opt
        self.use_pairwise=use_pairwise
        self.kwargs = kwargs

        self.results = {}
        self.figures = []
        self.model_datasets = []

        if self.feature_importance_opt is not None:
            self.optimize_for_feature_importance(self.feature_importance_opt)
            self.dataset = self.eval_dataset.copy() # this is a hot-fix solution        


    def _initial_sampler(self, initial_sample_size):
        """
        Randomly samples the initial points from the dataset.

        Returns:
            numpy.ndarray: Array of randomly selected points.
        """
        random_points = self.dataset.get_samples(initial_sample_size, remove_points=True)

        return random_points


    def _acquisition(self, model_dataset, add_noise: int = 0.1):
        """
        Performs the acquisition step to select new points for the model.

        Parameters:
            model: The model object used for acquisition.

        Returns:
            Dataset: The acquired dataset containing the selected points.
        """

        if self.acquisition_method == "greedy":

            preds = self.predict(self.dataset, self.model_dataset)

            # Find indices of the x-number of smallest values
            indices = np.argpartition(preds, self.acquisition_size)[:self.acquisition_size]

            # Get the best docked molecules from the dataset
            acq_dataset = self.dataset.get_points(indices, remove_points=True)

        if self.acquisition_method == "random":
            
            # Get random points and delete from dataset
            acq_dataset = self.dataset.get_samples(self.acquisition_size, remove_points=True)

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

            _, uncertainty = self.predict(self.dataset, self.model_dataset, return_uncertainty=True)
            # MU stands for most uncertainty.

            # Finds the indices with the highest uncertainty.
            indices = np.argpartition(uncertainty, -self.acquisition_size)[-self.acquisition_size:]

            acq_dataset = self.dataset.get_points(indices, remove_points=True)

        if self.acquisition_method == 'LCB':
            preds, uncertainty = self.predict(self.dataset, self.model_dataset, return_uncertainty=True)
            # LCB stands for Lower Confidence Bound.
            
            # Calculate the LCB score for each point.
            beta = 1  # This is a hyperparameter that can be tuned.
            lcb = preds - beta * uncertainty  # Note: Assuming lower preds are better.
            
            # Find the indices with the lowest LCB score.
            # Since np.argpartition finds indices for the smallest values and we're minimizing, it's directly applicable here.
            indices = np.argpartition(lcb, self.acquisition_size)[:self.acquisition_size]
            
            acq_dataset = self.dataset.get_points(indices, remove_points=True)

        if self.acquisition_method == "EI":
            preds, uncertainty = self.predict(self.dataset, self.model_dataset, return_uncertainty=True)
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
            preds, uncertainty = self.predict(self.dataset, self.model_dataset, return_uncertainty=True)
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


    def fit(self, iterations_in=None):

        if iterations_in is not None:
            feat_opt = True
        else:
            feat_opt = False

        # Seed handling
        if self.seeds is None or len(self.seeds) == 0:
            initial_pts = self._initial_sampler(initial_sample_size=self.initial_sample_size)
        elif isinstance(self.seeds, (list, np.ndarray)) and all(np.issubdtype(type(i), np.integer) for i in self.seeds):
            self.seeds = list(self.seeds)  # Ensure seeds is a list
            if feat_opt == True:
                initial_pts = self.dataset.get_points(self.seeds)
            else:
                initial_pts = self.dataset.get_points(self.seeds, remove_points=True)
        else:
            logging.error("Invalid seeds. Must be a list or ndarray of integers, or None.")
            return

        # Add noise to the initial points if desired
        if self.add_noise is None:
            self.model_dataset = initial_pts
        else:
            noises = np.random.normal(0, self.add_noise, size=initial_pts.y.size)
            initial_pts.y = initial_pts.y + noises
            self.model_dataset = initial_pts

        if not feat_opt:
            print(f"y values of starting points {initial_pts.y}")
        
        # fits the model using a pairwise dataset or normal dataset
        if self.use_pairwise:
            initial_pts_pairwise = initial_pts.create_pairwise_dataset()

            self.engine.fit(initial_pts_pairwise.X, initial_pts_pairwise.y)
        else:
            self.engine.fit(self.model_dataset.X, self.model_dataset.y)

        # First evaluation, using only the initial points
        if self.evaluator is not None and feat_opt is False:
            self.call_evaluator(i=-1, model_dataset=initial_pts) # -1 because ´call_evaluator´ starts at 1, and this iteration should be 0.

        # implemented to allow the ´fit´ method to be used internally in the class to support ´feature_importance_opt´.
        if iterations_in is None:
            iterations = self.iterations
        else:
            iterations = iterations_in

        for i in range(iterations):

            # Acquire new points
            acquired_pts = self._acquisition(model_dataset=self.model_dataset, add_noise=self.add_noise)

            self.model_dataset = self.dataset.merge_datasets([self.model_dataset, acquired_pts])

            # Reset model before training if true
            if self.retrain:
                self.engine = self.engine = Engine(self.engine_name, **self.kwargs)
            
            # fits the model using a pairwise dataset or normal dataset
            if self.use_pairwise:
                model_dataset_pairwise = self.model_dataset.create_pairwise_dataset()
                self.engine.fit(model_dataset_pairwise.X, model_dataset_pairwise.y)
            else:
                self.engine.fit(self.model_dataset.X, self.model_dataset.y)

            # Call evaluator if true
            if self.evaluator is not None and feat_opt is False:
                self.call_evaluator(i=i, model_dataset=self.model_dataset)

            if feat_opt:
                self._print_progress_bar(iteration=i, total=iterations)

        if feat_opt:
            print("\n")

        if self.model_graphs:
            self.graph_model()

        return self.engine


    def predict(self, dataset: Dataset, dataset_train: Dataset = None, return_uncertainty = False):

        if isinstance(dataset, Dataset) is False:
            logging.error("Wrong object type. Must be of type `Dataset`")
            sys.exit()

        if return_uncertainty:
            if self.use_pairwise:
                preds, uncertainty = self._pairwise_predict(dataset_train, dataset, self.engine)
            else:
                try:
                    preds, uncertainty = self.engine.predict(dataset.X)
                    if uncertainty is None:
                        raise NotImplementedError('Uncertainty is not implemented for this model.'
                                                  ' This probably means you cannot use this engine with the chosen acquisition functon.')
                except NotImplementedError as e:
                    raise e
        else:
            if self.use_pairwise:
                preds, _ = self._pairwise_predict(dataset_train, dataset, self.engine)
            else:
                preds, _ = self.engine.predict(dataset.X, no_uncertainty=True)
        
        if return_uncertainty:
            return preds, uncertainty
        else:
            return preds

    # Old PADRE implementation
    # def _pairwise_predict(self, train_dataset: Dataset, predict_dataset: Dataset, engine: Engine):

    #     # Split up prediction if dataset size is greater than 10k.
    #     dataset_size = predict_dataset.y.shape[0]
    #     batch_size = 10000

    #     if dataset_size > batch_size:

    #         # Split prediction dataset into batches of 10k and less for the last one.
    #         split_datasets = []
    #         for i in range(0, dataset_size, batch_size):
    #             end_index = min(i + batch_size, dataset_size)
    #             indices = list(range(i, end_index))
    #             batch_dataset = predict_dataset.get_points(indices)
    #             split_datasets.append(batch_dataset)

    #         # Predict on each of the prediction datasets
    #         n2 = train_dataset.X.shape[0]

    #         mu_list = []
    #         std_list = []
    #         for split_set in split_datasets:
    #             n1 = split_set.X.shape[0]
    #             X1X2 = self.PADRE_features(split_set.X, train_dataset.X)
    #             y1_minus_y2_hat = engine.predict(X1X2)[0]
    #             y1_hat_distribution = y1_minus_y2_hat.reshape(n1, n2) + train_dataset.y[np.newaxis, :]
    #             mu = y1_hat_distribution.mean(axis=1)
    #             std = y1_hat_distribution.std(axis=1)
    #             mu_list.append(mu)
    #             std_list.append(std)

    #         mu = np.concatenate(mu_list)
    #         std = np.concatenate(std)

    #     else:
    #         n1 = predict_dataset.X.shape[0]
    #         n2 = train_dataset.X.shape[0]

    #         X1X2 = self.PADRE_features(predict_dataset.X, train_dataset.X)
    #         y1_minus_y2_hat = engine.predict(X1X2)[0]
    #         y1_hat_distribution = y1_minus_y2_hat.reshape(n1, n2) + train_dataset.y[np.newaxis, :]
    #         mu = y1_hat_distribution.mean(axis=1)
    #         std = y1_hat_distribution.std(axis=1)

    #     return mu, std

# New PADRE implementation
# --------------------------------
    def parallel_predict_chunk(self, start_idx, end_idx, engine, train_dataset, predict_dataset):
        X_chunk = predict_dataset.X[start_idx:end_idx]
        return self._pairwise_predict_chunk(engine, train_dataset, X_chunk)

    def _pairwise_predict_chunk(self, engine, train_dataset, X_chunk):
        n1 = X_chunk.shape[0]
        n2 = train_dataset.X.shape[0]

        X1X2 = self.PADRE_features(X_chunk, train_dataset.X)
        y1_minus_y2_hat = engine.predict(X1X2, no_uncertainty=True)[0]
        y1_hat_distribution = y1_minus_y2_hat.reshape(n1, n2) + train_dataset.y[np.newaxis, :]
        mu = y1_hat_distribution.mean(axis=1)
        std = y1_hat_distribution.std(axis=1)
        return mu, std
    
    def _pairwise_predict(self, train_dataset, predict_dataset, engine):
        dataset_size = predict_dataset.X.shape[0]
        chunk_size = 1000  # Maximum chunk size to avoid memory issues

        n_chunks = max(1, (dataset_size + chunk_size - 1) // chunk_size)

        if n_chunks < num_cores:
            chunk_size = dataset_size // num_cores
            n_chunks = num_cores

        with parallel_config('threading'):
            results = Parallel(n_jobs=num_cores)(
                delayed(self.parallel_predict_chunk)(i, min(i + chunk_size, dataset_size), engine, train_dataset, predict_dataset)
                for i in range(0, dataset_size, chunk_size)
            )

        mu_list, std_list = zip(*results)
        mu = np.concatenate(mu_list)
        std = np.concatenate(std_list)

        return mu, std
# --------------------------------

    def optimize_for_feature_importance(self, opt_parameters: Dict):

        print('Computing feature importance...')
        if self.engine_name != 'RF':
            print('Feature optimization tests has only been implemented for random forest (RF)')
            print('Terminating program...')
            sys.exit()

        iterations = opt_parameters['iterations']
        features_limit = opt_parameters['features_limit']
    
        self.fit(iterations_in=iterations)
        engine = self.engine.access_engine()

        feature_importances = engine.feature_importances_
        feature_importances_sorted = np.argsort(feature_importances)[:-1]
        important_features = feature_importances_sorted[-features_limit:]

        self.dataset.X = self.dataset.X[:, important_features]
        self.eval_dataset.X = self.eval_dataset.X[:, important_features]

        # important_feature_values = feature_importances[important_features]
        # print(f"values of most important features: {important_feature_values}")
        
        print(f"Indices of most important features: {important_features} \n")

        return important_features


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


    def save(self, filename: str):
        """
        Save the RFModeller to a pickle file
        """
        # Check if filename is a string.
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
        except FileNotFoundError:
            logging.error(f"File not found: {filename}")
            raise
        except IOError as e:
            logging.error(f"IOError: {str(e)}")
            raise
        except pickle.PicklingError as e:
            logging.error(f"Failed to pickle model: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise


    @staticmethod
    def load(filename: str):
        
        # Check if filename is a string.
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        
        # Check if file exists.
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No such file or directory: '{filename}'")
        
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            logging.error(f"File not found: {filename}")
            raise
        except IOError as e:
            logging.error(f"IOError: {str(e)}")
            raise
        except pickle.UnpicklingError as e:
            logging.error(f"Failed to unpickle model: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise


    def _print_progress_bar(self, iteration, total, bar_length=50, prefix="Progress"):
        """
        Print the progress bar.

        Args:
            iteration (int): current iteration.
            total (int): total iterations.
            bar_length (int): length of the progress bar.
            prefix (str): Prefix to print before the progress bar. Default is "Progress".
        """
        iteration = iteration + 1
        progress = (iteration / total)
        arrow = '-' * int(round(progress * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write(f"\r{prefix}: [{arrow + spaces}] {int(progress * 100)}% ({iteration}/{total})")
        sys.stdout.flush()


    def PADRE_features(self, X1, X2):

        n1 = X1.shape[0]
        n2 = X2.shape[0]

        X1 = X1[:, newaxis, :].repeat(n2, axis=1)
        X2 = X2[newaxis, :, :].repeat(n1, axis=0)

        X1X2_combined = concatenate([X1, X2, X1 - X2], axis=2)
        return X1X2_combined.reshape(n1 * n2, -1)


    def PADRE_labels(self, y1, y2):
        return (y1[:, newaxis] - y2[newaxis, :]).flatten()


    def PADRE_train(self, model, train_X, train_y):
        X1X2 = self.PADRE_features(train_X, train_X)
        y1_minus_y2 = self.PADRE_labels(train_y, train_y)
        model.fit(X1X2, y1_minus_y2)
        return model


    def PADRE_predict(self, model, test_X, train_X, train_y):
        n1 = test_X.shape[0]
        n2 = train_X.shape[0]

        X1X2 = self.PADRE_features(test_X, train_X)
        y1_minus_y2_hat = model.predict(X1X2)
        y1_hat_distribution = y1_minus_y2_hat.reshape(n1, n2) + train_y[newaxis, :]
        mu = y1_hat_distribution.mean(axis=1)
        std = y1_hat_distribution.std(axis=1)
        return mu, std