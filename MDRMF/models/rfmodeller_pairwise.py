import logging
import pickle
import os
import sys
from typing import Dict
import numpy as np
from numpy import newaxis, concatenate, std
from sklearn.ensemble import RandomForestRegressor
from MDRMF.models.modeller import Modeller
from MDRMF.dataset import Dataset

class RFModellerPairwise(Modeller):

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
        model_graphs=False,
        feature_importance_opt=None,
        use_pairwise=True,
        **kwargs) -> None:

        super().__init__(
            dataset, 
            evaluator,
            iterations, 
            initial_sample_size, 
            acquisition_size, 
            acquisition_method, 
            retrain,
            seeds,
            model_graphs
            )

        self.kwargs = kwargs
        self.model = RandomForestRegressor(**self.kwargs)
        self.feature_importance_opt = feature_importance_opt
        self.use_pairwise = use_pairwise
        self.model_dataset = None

        if self.feature_importance_opt is not None:
            self.optimize_for_feature_importance(self.feature_importance_opt)
            self.dataset = self.eval_dataset.copy() # this is a hot-fix solution

    def fit(self, iterations_in=None):

        if iterations_in is not None:
            feat_opt = True
        else:
            feat_opt = False

        # Seeds handling
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

        self.model_dataset = initial_pts # hot-fix solution.
        '''
        It is evidents from the above that we can rewrite the merging code,
        so it only needs to merge self.model_dataset and acquired_pts.
        '''

        if not feat_opt:
            print(f"y values of starting points {initial_pts.y}")
        
        if self.use_pairwise:
            initial_pts_pairwise = initial_pts.create_pairwise_dataset()

            self.model.fit(initial_pts_pairwise.X, initial_pts_pairwise.y)
        else:
            self.model.fit(initial_pts.X, initial_pts.y)

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
            acquired_pts = self._acquisition_pairwise()

            # Merge old and new points
            if i == 0:
                self.model_dataset = self.dataset.merge_datasets([initial_pts, acquired_pts])
            else:
                self.model_dataset = self.dataset.merge_datasets([self.model_dataset, acquired_pts])

            # Reset model before training if true
            if self.retrain:
                self.model = RandomForestRegressor(**self.kwargs)
            
            # fits the model using a pairwise dataset or normal dataset
            if self.use_pairwise:
                model_dataset_pairwise = self.model_dataset.create_pairwise_dataset()
                self.model.fit(model_dataset_pairwise.X, model_dataset_pairwise.y)
            else:
                self.model.fit(self.model_dataset.X, self.model_dataset.y)

            # Call evaluator if true
            if self.evaluator is not None and feat_opt is False:
                self.call_evaluator(i=i, model_dataset=self.model_dataset)

            if feat_opt:
                self._print_progress_bar(iteration=i, total=iterations)

        if feat_opt:
            print("\n")

        return self.model
    

    def _pairwise_predict_old(self, train_dataset: Dataset, predict_dataset: Dataset, model: RandomForestRegressor):
        """
        Predicts molecular properties in the prediction dataset using a RandomForestRegressor model and pairwise comparisons with the training dataset.

        Parameters:
        - train_dataset (Dataset): Dataset with known molecular properties for training.
        - predict_dataset (Dataset): Dataset of molecules to predict properties for.
        - model (RandomForestRegressor): Trained RandomForestRegressor model.

        Returns:
        - List[float]: Predicted properties for each molecule in the predict_dataset.
        """
        mols_predict = predict_dataset.X
        mols_train = train_dataset.X
        y_train = train_dataset.y

        predictions = []
        for pmol in mols_predict:
            X_test = []
            for tmol in mols_train:
                X = list(tmol) + list(pmol) + (list(tmol - pmol))
                X_test.append(X)      
            
            dy_preds = model.predict(np.array(X_test))
            y_preds = y_train + dy_preds
            y_pred = y_preds.mean()
            predictions.append(y_pred)

        predictions = np.array(predictions)
        return predictions

    
    def _pairwise_predict(self, train_dataset: Dataset, predict_dataset: Dataset, model: RandomForestRegressor):
        
        # Split up prediction if dataset size is greater than 10k.
        dataset_size = predict_dataset.y.shape[0]
        batch_size = 10000

        if dataset_size > batch_size:
            
            # Split prediction dataset into batches of 10k and less for the last one.
            split_datasets = []
            for i in range(0, dataset_size, batch_size):
                end_index = min(i + batch_size, dataset_size)
                indices = list(range(i, end_index))
                batch_dataset = predict_dataset.get_points(indices)
                split_datasets.append(batch_dataset)

            # Predict on each of the prediction datasets
            n2 = train_dataset.X.shape[0]

            mu_list = []
            for split_set in split_datasets:
                n1 = split_set.X.shape[0]
                X1X2 = self.PADRE_features(split_set.X, train_dataset.X)
                y1_minus_y2_hat = model.predict(X1X2)
                y1_hat_distribution = y1_minus_y2_hat.reshape(n1, n2) + train_dataset.y[np.newaxis, :]
                mu = y1_hat_distribution.mean(axis=1)
                mu_list.append(mu)

            mu = np.concatenate(mu_list)

        else:
            n1 = predict_dataset.X.shape[0]
            n2 = train_dataset.X.shape[0]

            X1X2 = self.PADRE_features(predict_dataset.X, train_dataset.X)
            y1_minus_y2_hat = model.predict(X1X2)
            y1_hat_distribution = y1_minus_y2_hat.reshape(n1, n2) + train_dataset.y[np.newaxis, :]
            mu = y1_hat_distribution.mean(axis=1)
            #std = y1_hat_distribution.std(axis=1)
        
        return mu
    

    def predict(self, dataset: Dataset, dataset_train: Dataset = None):

        if isinstance(dataset, Dataset) is False:
            logging.error("Wrong object type. Must be of type `Dataset`")
            sys.exit()
            
        if isinstance(dataset, Dataset) and isinstance(dataset_train, Dataset) and self.use_pairwise:
            preds = self._pairwise_predict(dataset_train, dataset, self.model)

        elif isinstance(dataset, Dataset) is True:
            preds = self.model.predict(dataset.X)

        return preds


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
    

    def _acquisition_pairwise(self):
        
        preds = self.predict(self.dataset, self.model_dataset)

        if self.acquisition_method == "greedy":

            # Find indices of the x-number of smallest values
            indices = np.argpartition(preds, self.acquisition_size)[:self.acquisition_size]

            # Get the best docked molecules from the dataset
            acq_dataset = self.dataset.get_points(indices, remove_points=True)
            print(acq_dataset.y)

        if self.acquisition_method == "random":
            
            # Get random points and delete from dataset
            acq_dataset = self.dataset.get_samples(self.acquisition_size, remove_points=True)

        return acq_dataset


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


    def optimize_for_feature_importance(self, opt_parameters: Dict):

        print('Computing feature importance...')

        iterations = opt_parameters['iterations']
        features_limit = opt_parameters['features_limit']
    
        model = self.fit(iterations_in=iterations)

        feature_importances = model.feature_importances_
        feature_importances_sorted = np.argsort(feature_importances)[:-1]
        important_features = feature_importances_sorted[-features_limit:]

        self.dataset.X = self.dataset.X[:, important_features]
        self.eval_dataset.X = self.eval_dataset.X[:, important_features]

        # important_feature_values = feature_importances[important_features]
        # print(f"values of most important features: {important_feature_values}")
        
        print(f"Indices of most important features: {important_features} \n")

        return important_features

        # --- Comments
        # There should be an argument to RFModeller called ´feature_importance_opt´.
        # This argument should take a dict.
        # dict = {
        #   'opt_iterations': 50
        #   'opt_features_limit': 30
        # }
        # This list contain the number of times to train the optimization model and how many of the most important features to keep.
        # For each run the ´feature_importances_´ is calculated. The index values are counted. So for instance
        # 1. run index 55 is the most important feature
        # 2. run index 55 is the 10th most important feature
        # 3. run index 55 is the 5th most important feature
        # Now we just average how much 55 was used. (1+10+5)/3 = 5.33
        # We then just calculate an average for each index in the vector and sort them from lowest(best) to highest(worst).
        # In the case of the above dict we only keep the 30 most important features in the vector.
        # -----
        # Now that I think about it we might not even need to do this averaging, as the numbers are kind of already
        # averaged by merely running the model many times.
        # -----
        # Once the desired features have been found we need to set he dataset.X to the indexes that was found most important.
        # I think we can do this by just manipulating self.dataset, but I am a little unsure if this will disturb other parts
        # of the code. I don't think so, as we never return the Dataset at any time.

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

# PADRE functions
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