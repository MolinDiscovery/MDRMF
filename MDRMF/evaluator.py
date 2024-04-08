import numpy as np
from sklearn.metrics import r2_score
import copy


class Evaluator:
    
    def __init__(self, original_dataset, metrics, k_values):
        self.dataset = copy.deepcopy(original_dataset)
        self.metrics = metrics
        self.k_values = [int(k) for k in k_values]


    def evaluate(self, model, eval_dataset, model_dataset):
        self.dataset = eval_dataset.copy()
        results = {}
        for metric in self.metrics:
            if metric == "R2_model":
                results[metric] = self.r2_model(model, eval_dataset)
            else:
                for k in self.k_values:
                    if metric == "top-k":
                        results[f"top-{k} model"] = self.top_n_correct(k, model, model_dataset)
                    elif metric == "R2_k":
                        results[f"R2_k-{k}"] = self.r2_n(k, model, model_dataset)
                    elif metric == "top-k-acquired":
                        results[f"top-{k} acquired"] = self.top_n_in_model_set(k, model_dataset)
        return results


    def top_n_correct(self, n, model, model_dataset):
        model_predictions = model.predict(self.dataset, model_dataset) # Predict on the full dataset
        preds_indices = np.argsort(model_predictions)[:n] # Sort all predictions from lowest to highest and gets the indices of n amount of mols
        top_n_real_indices = np.argsort(self.dataset.y)[:n] # Get the indices of the n "real" mols and sorts them from lowest to highest
        return np.mean(np.isin(preds_indices, top_n_real_indices)) # np.isin calculates how many from the correct_preds_indices that are in top_n_real_indices and np.mean makes this a fraction
    

    def top_n_in_model_set(self, n, model_dataset):
        lowest_y_indices = np.argsort(self.dataset.y)[:n]  # Get indices of the 'n' lowest y values.
        lowest_y_ids = set(self.dataset.ids[lowest_y_indices])  # Retrieve corresponding IDs from the dataset and ensure uniqueness.

        ids_acquired = set(model_dataset.ids)  # Retrieve unique ids from the internal model dataset.
        intersection_count = len(lowest_y_ids.intersection(ids_acquired))  # Count of common ids between lowest_y_ids and ids_acquired.
        
        return intersection_count / n  # Return the proportion of top 'n' found in the model_dataset.  


    def r2_model(self, model, model_dataset):
        '''
        Returns the R2 value of the internal model
        '''

        # Find missing points in the model_dataset
        training_points = self.dataset.missing_points(self.dataset, model_dataset)

        y_true = training_points.y
        y_pred = model.predict(training_points)

        return r2_score(y_true, y_pred)
    

    def r2_n(self, n, model, model_dataset):
        # Similar to top_n_correct but here we calculate the r2 score for the top n points
        model_predictions = model.predict(self.dataset, model_dataset)
        top_n_pred_indices = np.argsort(model_predictions)[:n]

        # Get top n points as a Dataset
        top_n_dataset = self.dataset.get_points(top_n_pred_indices)

        y_pred = model.predict(top_n_dataset)
        
        return r2_score(top_n_dataset.y, y_pred)