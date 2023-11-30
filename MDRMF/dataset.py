# dataset.py

from re import U
import numpy as np
import pickle


class Dataset:

    def __init__(self, X, y, ids=None, w=None, keep_unlabeled_data_only=False) -> None:
        
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

        self.X = X
        self.y = y
        self.ids = ids
        self.w = w

        # Validate the input data
        if not all(len(data) == len(self.X) for data in [self.y, self.ids, self.w]):
            raise ValueError("Inconsistent input data: all input data should have the same number of samples.")

        # Remove potential NaN values.         
        if keep_unlabeled_data_only == False:
            self.remove_invalid_entries()

        # If the data is semi-labeled
        # the unlabeled points should be preserved.
        if keep_unlabeled_data_only == True:
            self.keep_unlabel_entries_only()
        

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

    def get_points(self, indices, remove_points=False, unlabeled=False):

        '''
        Retrieves specific points from the dataset based on the provided list of indices. 
        This method can optionally remove these points from the dataset after retrieval.

        Parameters:
            indices (list): A list of indices specifying the data points to retrieve from the dataset.
            remove_points (bool, optional): If set to True, the data points corresponding to the provided indices will be removed from the dataset. Defaults to False.
            unlabeled (bool, optional): If set to True the retrieved dataset will not have NaN values removed.

        Returns:
            Dataset: A new Dataset object containing the data points specified by the indices. This dataset includes the features (`X`), labels/targets (`y`), identifiers (`ids`), and any additional weights (`w`) associated with these data points.

        Note:
            The removal of points from the dataset is an in-place operation. If `remove_points` is set to True, the original dataset will be modified.
        '''

        g_X = self.X[indices]
        g_y = self.y[indices]
        g_ids = self.ids[indices]
        g_w = self.w[indices]

        if remove_points:
            self.remove_points(indices)

        if unlabeled == True:
            return Dataset(g_X, g_y, g_ids, g_w, keep_unlabeled_data_only=True)
        else:
            return Dataset(g_X, g_y, g_ids, g_w)

    def get_samples(self, n_samples, remove_points=False, return_indices=False, unlabeled=False):
        random_indices = np.random.choice(len(self.X), size=n_samples, replace=False)
        g_X = self.X[random_indices]
        g_y = self.y[random_indices]
        g_ids = self.ids[random_indices]
        g_w = self.w[random_indices]

        if unlabeled == True:
            sampled_dataset = Dataset(g_X, g_y, g_ids, g_w, keep_unlabeled_data_only=True)
        else:
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
    
    
    def keep_unlabel_entries_only(self):
        '''
        Keep only entries in the dataset where y is NaN
        '''
        unlabeled_data_indices = np.where(np.isnan(self.y))[0]

        self.X = self.X[unlabeled_data_indices]
        self.y = self.y[unlabeled_data_indices]
        self.ids = self.ids[unlabeled_data_indices]
        self.w = self.w[unlabeled_data_indices]
    
    @staticmethod
    def remove_mismatched_ids(*datasets):
        """
        Efficiently compares multiple Dataset instances and removes entries with non-identical IDs across them.
        Parameters:
        *datasets : a variable number of Dataset instances to be compared.
        Returns:
        A tuple of Dataset instances with mismatched IDs removed.
        """
        # Create sets of IDs from each dataset for fast intersection
        ids_sets = [set(dataset.ids) for dataset in datasets]
        
        # Find the common IDs by intersecting the sets
        common_ids = set.intersection(*ids_sets)
        
        # Convert the common IDs back to a sorted NumPy array for indexing
        common_ids = np.array(sorted(common_ids), dtype=datasets[0].ids.dtype)
        
        # Filter each dataset to only include entries with IDs in the common set
        filtered_datasets = []
        for dataset in datasets:
            # Create a boolean index mask for the current dataset's IDs
            mask = np.isin(dataset.ids, common_ids)
            
            # Filter the dataset using the boolean index mask
            filtered_dataset = Dataset(
                X=dataset.X[mask],
                y=dataset.y[mask],
                ids=dataset.ids[mask],
                w=dataset.w[mask]
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
    

    def check_mismatches(self, *datasets):
        """
        Efficiently compares multiple Dataset instances and identifies entries with non-identical IDs across them.
        Parameters:
        *datasets : a variable number of Dataset instances to be compared.
        Returns:
        A dictionary with the mismatched IDs for each dataset.
        """
        mismatches = {}
        dataset_ids_sets = [set(dataset.ids) for dataset in datasets]  # Convert IDs to sets for O(1) lookups
        all_ids_set = set().union(*dataset_ids_sets)  # Union of all IDs

        # Loop over each dataset and compare against the union of all IDs
        for i, dataset_ids in enumerate(dataset_ids_sets):
            mismatched_ids = all_ids_set - dataset_ids  # Set difference to find mismatches

            # Store only the mismatched IDs
            mismatches[f"dataset_{i}"] = sorted(mismatched_ids)  # Sort for consistent ordering

        return mismatches

