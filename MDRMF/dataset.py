# dataset.py

import numpy as np
import pickle


class Dataset:

    def __init__(self, X, y, ids=None, w=None) -> None:
        
        # Convert inputs to NumPy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        ids = np.arange(len(X)) if ids is None else np.asarray(ids, dtype=object)
        w = np.ones(len(X), dtype=np.float32) if w is None else np.asarray(w)

        # Check if X needs stacking
        if X.ndim == 1 and isinstance(X[0], (np.ndarray, list)):
            try:
                X = np.stack(X)
            except ValueError as e:
                raise ValueError("X should be a 2D array-like structure with consistent inner dimensions.") from e

        # n_samples = np.shape(X)[0]

        # if w is None:
        #     if len(y.shape) == 1:
        #         w = np.ones(y.shape[0], np.float32)
        #     else:
        #         w = np.ones((y.shape[0], 1), np.float32)

        # if ids is None:
        #     ids = np.arange(n_samples)

        self.X = X
        self.y = y
        self.ids = ids
        self.w = w

        # Validate the input data
        if not all(len(data) == len(self.X) for data in [self.y, self.ids, self.w]):
            raise ValueError("Inconsistent input data: all input data should have the same number of samples.")
        
        # Remove potential NaN values
        self.remove_invalid_entries()

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


    def remove_invalid_entries(self):
        """
        Remove rows from the dataset where either X or y contains NaNs.
        """
        # Find indices where X or y contains NaNs
        invalid_indices_x = np.where(np.isnan(self.X).any(axis=1))[0]
        invalid_indices_y = np.where(np.isnan(self.y))[0]

        # Combine the indices and remove duplicates
        invalid_indices = np.unique(np.concatenate((invalid_indices_x, invalid_indices_y)))

        # Remove these points from the dataset using the existing method
        self.remove_points(invalid_indices)


    @staticmethod
    def remove_mismatched_ids(*datasets):
        """
        Compares multiple Dataset instances and removes entries with non-identical IDs across them.
        Parameters:
        *datasets : a variable number of Dataset instances to be compared.
        Returns:
        A tuple of Dataset instances with mismatched IDs removed.
        """
        # Find the intersection of IDs across all datasets using NumPy for efficiency
        common_ids = datasets[0].ids
        for dataset in datasets[1:]:
            common_ids = np.intersect1d(common_ids, dataset.ids, assume_unique=True)

        # Filter each dataset to only include entries with IDs in the intersection
        filtered_datasets = []
        for dataset in datasets:
            indices_to_keep = np.isin(dataset.ids, common_ids)
            filtered_dataset = Dataset(
                X=np.ascontiguousarray(dataset.X[indices_to_keep]),
                y=np.ascontiguousarray(dataset.y[indices_to_keep]),
                ids=np.ascontiguousarray(dataset.ids[indices_to_keep]),
                w=np.ascontiguousarray(dataset.w[indices_to_keep])
            )
            filtered_datasets.append(filtered_dataset)

        return tuple(filtered_datasets)
    

    @staticmethod
    def check_ids_order(*datasets):
        """
        Checks if all provided datasets have the same ids in the same order.

        Parameters:
        *datasets : a variable number of Dataset instances to be compared.

        Returns:
        bool: True if all ids match and are in the same order, False otherwise.
        """
        # We can skip the check if there's only one or no datasets
        if len(datasets) < 2:
            return True

        # Use the ids of the first dataset as the reference
        reference_ids = datasets[0].ids

        # Check each subsequent dataset against the reference
        for dataset in datasets[1:]:
            if not np.array_equal(reference_ids, dataset.ids):
                return False  # Found a dataset with different ids or order

        # All datasets have the same ids in the same order
        return True    