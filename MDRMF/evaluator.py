import numpy as np
from sklearn.metrics import r2_score
import copy


class Evaluator:
    def __init__(self, original_dataset, metrics, k_values):
        self.dataset = copy.deepcopy(original_dataset)
        self.metrics = metrics
        self.k_values = [int(k) for k in k_values]

    def evaluate(self, model, model_dataset):
        results = {}
        for metric in self.metrics:
            if metric == "R2_model":
                results[metric] = self.r2_model(model, model_dataset)
            else:
                for k in self.k_values:
                    if metric == "top-k":
                        results[f"top-{k}"] = self.top_n_correct(k, model)
                    elif metric == "R2_k":
                        results[f"R2_k-{k}"] = self.r2_n(k, model)
        return results

    def top_n_correct(self, n, model):
        model_predictions = model.predict(self.dataset)
        correct_preds_indices = np.argsort(model_predictions)[:n]
        top_n_real_indices = np.argsort(self.dataset.y)[:n]
        return np.mean(np.isin(correct_preds_indices, top_n_real_indices))


    def r2_model(self, model, model_dataset):
        '''
        Returns the R2 value of the internal model
        '''

        # Find missing points in the model_dataset
        training_points = self.dataset.missing_points(self.dataset, model_dataset)

        y_true = training_points.y
        y_pred = model.predict(training_points)

        return r2_score(y_true, y_pred)
    

    def r2_n(self, n, model):
        # Similar to top_n_correct but here we calculate the r2 score for the top n points
        model_predictions = model.predict(self.dataset)
        top_n_pred_indices = np.argsort(model_predictions)[:n]

        # Get top n points as a Dataset
        top_n_dataset = self.dataset.get_points(top_n_pred_indices)

        y_pred = model.predict(top_n_dataset)
        
        return r2_score(top_n_dataset.y, y_pred)

