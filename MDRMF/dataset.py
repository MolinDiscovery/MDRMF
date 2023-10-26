# dataset.py

import numpy as np
import pickle

class Dataset:

    def __init__(self, X, y, ids=None, w=None) -> None:
        
        n_samples = np.shape(X)[0]

        if w is None:
            if len(y.shape) == 1:
                w = np.ones(y.shape[0], np.float32)
            else:
                w = np.ones((y.shape[0], 1), np.float32)

        if ids is None:
            ids = np.arange(n_samples)

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.ids = np.array(ids, dtype=object)
        self.w = np.asarray(w)

        # Check consistency of input data
        if not all(len(data) == n_samples for data in [self.X, self.y, self.ids, self.w]):
            raise ValueError("Inconsistent input data: all input data should have the same number of samples.")

    def __repr__(self):
        return f"<Dataset X.shape: {self.X.shape}, y.shape: {self.y.shape}, w.shape: {self.w.shape}, ids: {self.ids}>"
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    def get_length(self):
        return len(self.w)

    def get_points(self, indices, remove_points=False):
        g_X = self.X[indices]
        g_y = self.y[indices]
        g_ids = self.ids[indices]
        g_w = self.w[indices]

        if remove_points:
            self.remove_points(indices)

        return Dataset(g_X, g_y, g_ids, g_w)

    def get_samples(self, n_samples, remove_points=False, return_indices=False):
        random_indices = np.random.choice(len(self.X), size=n_samples, replace=False)
        g_X = self.X[random_indices]
        g_y = self.y[random_indices]
        g_ids = self.ids[random_indices]
        g_w = self.w[random_indices]

        sampled_dataset = Dataset(g_X, g_y, g_ids, g_w)

        if remove_points:
            self.remove_points(random_indices)
        
        if return_indices:
            return sampled_dataset, random_indices
        else:
            return sampled_dataset

    def set_points(self, indices):
        self.X = self.X[indices]
        self.y = self.y[indices]
        self.ids = self.ids[indices]
        self.w = self.w[indices]

    def remove_points(self, indices):
        indices = np.sort(indices)[::-1] # remove indices from desending order
        mask = np.ones(len(self.X), dtype=bool)
        mask[indices] = False
        self.X = self.X[mask]
        self.y = self.y[mask]
        self.ids = self.ids[mask]
        self.w = self.w[mask]

    def sort_by_y(self, ascending=True):
        sort_indices = np.argsort(self.y)

        if not ascending:
            sort_indices = sort_indices[::-1]

        self.X = self.X[sort_indices]
        self.y = self.y[sort_indices]
        self.ids = self.ids[sort_indices]
        self.w = self.w[sort_indices]

    @staticmethod
    def merge_datasets(datasets):
        # Initialize empty lists for X, y, ids, and w
        X, y, ids, w = [], [], [], []

        # Loop over the datasets
        for dataset in datasets:
            # Append the data from each dataset to the corresponding list
            X.append(dataset.X)
            y.append(dataset.y)
            ids.append(dataset.ids)
            w.append(dataset.w)

        # Convert lists to numpy arrays and concatenate along the first axis
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        ids = np.concatenate(ids, axis=0)
        w = np.concatenate(w, axis=0)

        # Return a new Dataset that combines the data from all the datasets
        return Dataset(X, y, ids, w)
    
    @staticmethod
    def missing_points(original_dataset, model_dataset):
        # compare the ids

        points_in_model = np.isin(original_dataset.ids, model_dataset.ids, invert=True)
        dataset = original_dataset.get_points(points_in_model)

        return dataset
    
    def copy(self):
        import copy
        return copy.deepcopy(self)
