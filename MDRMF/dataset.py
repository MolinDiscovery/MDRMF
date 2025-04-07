"""
Dataset module.

This module provides the Dataset class for handling and manipulating datasets in machine learning tasks.
It offers functionalities for splitting, shuffling, merging, sampling, and other dataset operations.
All methods are documented using Google style docstrings.
"""

from re import U  # Unused import, consider removing if not required
import numpy as np
import pickle


class Dataset:
    """
    A class to represent and manipulate a dataset.

    Attributes:
        X (np.ndarray): Feature data.
        y (np.ndarray): Labels or targets.
        ids (np.ndarray): Unique identifiers for each sample.
        w (np.ndarray): Weights for each sample.
    """

    def __init__(self, X, y, ids=None, w=None, keep_unlabeled_data_only=False) -> None:
        """
        Initialize the Dataset object.

        This constructor converts input data to NumPy arrays, validates their dimensions, and optionally removes
        invalid entries or retains only unlabeled data.

        Args:
            X (array-like): Feature data.
            y (array-like): Labels corresponding to each sample.
            ids (array-like, optional): Unique identifiers for each sample. If None, sequential integers are assigned.
            w (array-like, optional): Weights for each sample. If None, all samples are given a weight of 1.
            keep_unlabeled_data_only (bool, optional): If True, only retains data points with NaN labels.
                Otherwise, removes entries with NaN values in X or y. Defaults to False.

        Raises:
            ValueError: If X cannot be stacked into a 2D array-like structure with consistent inner dimensions.
            ValueError: If the input data arrays have inconsistent numbers of samples.
        """
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

        # Check that all input arrays have the same length
        self.X = X
        self.y = y
        self.ids = ids
        self.w = w

        if not all(len(data) == len(self.X) for data in [self.y, self.ids, self.w]):
            raise ValueError("Inconsistent input data: all input data should have the same number of samples.")

        # Remove potential NaN values or keep only unlabeled data
        if keep_unlabeled_data_only is False:
            self.remove_invalid_entries()
        if keep_unlabeled_data_only is True:
            self.keep_unlabel_entries_only()

    def __repr__(self):
        """
        Return a string representation of the Dataset.

        Returns:
            str: A string showing the shapes of X, y, w, and the ids.
        """
        return f"<Dataset X.shape: {self.X.shape}, y.shape: {self.y.shape}, w.shape: {self.w.shape}, ids: {self.ids}>"

    def save(self, filename):
        """
        Save the Dataset object to a file using pickle.

        Args:
            filename (str): The file path where the Dataset object will be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load a Dataset object from a file.

        Args:
            filename (str): The file path from which to load the Dataset object.

        Returns:
            Dataset: The loaded Dataset object.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def get_length(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.w)

    def get_points(self, indices, remove_points=False, unlabeled=False):
        """
        Retrieve specific points from the dataset based on provided indices.

        Args:
            indices (list or array-like): List of indices specifying the data points to retrieve.
            remove_points (bool, optional): If True, removes the data points from the original dataset. Defaults to False.
            unlabeled (bool, optional): If True, returns a dataset where NaN values are not removed. Defaults to False.

        Returns:
            Dataset: A new Dataset object containing the selected data points.

        Note:
            If `remove_points` is True, the removal is performed in-place on the original dataset.
        """
        g_X = self.X[indices]
        g_y = self.y[indices]
        g_ids = self.ids[indices]
        g_w = self.w[indices]

        if remove_points:
            self.remove_points(indices)

        if unlabeled:
            return Dataset(g_X, g_y, g_ids, g_w, keep_unlabeled_data_only=True)
        else:
            return Dataset(g_X, g_y, g_ids, g_w)

    def get_points_from_ids(self, ids: list):
        """
        Retrieve data points based on a list of identifiers.

        Args:
            ids (list): List of identifiers for the data points.

        Returns:
            Dataset: A new Dataset object containing data points with the specified identifiers.
        """
        indices = np.where(np.isin(self.ids, ids))[0]
        return self.get_points(indices)

    def get_indices_from_ids(self, ids: list):
        """
        Retrieve indices for data points based on a list of identifiers.

        Args:
            ids (list): List of identifiers for the data points.

        Returns:
            np.ndarray: Array of indices corresponding to the provided identifiers.
        """
        return np.where(np.isin(self.ids, ids))[0]

    def get_samples(self, n_samples, remove_points=False, return_indices=False, unlabeled=False):
        """
        Randomly sample a subset of data points from the dataset.

        Args:
            n_samples (int): Number of samples to retrieve.
            remove_points (bool, optional): If True, removes the sampled points from the original dataset. Defaults to False.
            return_indices (bool, optional): If True, returns a tuple of the sampled Dataset and the indices of the sampled points.
                Defaults to False.
            unlabeled (bool, optional): If True, retains NaN values in labels. Defaults to False.

        Returns:
            Dataset or tuple: A new Dataset object containing the sampled data points, or a tuple (Dataset, indices)
            if return_indices is True.
        """
        random_indices = np.random.choice(len(self.y), size=n_samples, replace=False)
        g_X = self.X[random_indices]
        g_y = self.y[random_indices]
        g_ids = self.ids[random_indices]
        g_w = self.w[random_indices]

        if unlabeled:
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
        """
        Update the dataset to only include data points at the specified indices.

        Args:
            indices (list or array-like): Indices of the data points to retain.
        """
        self.X = self.X[indices]
        self.y = self.y[indices]
        self.ids = self.ids[indices]
        self.w = self.w[indices]

    def remove_points(self, indices):
        """
        Remove data points from the dataset at the specified indices.

        Args:
            indices (list or array-like): Indices of the data points to remove.
        """
        indices = np.sort(indices)[::-1]  # Remove indices in descending order
        mask = np.ones(len(self.X), dtype=bool)
        mask[indices] = False
        self.X = self.X[mask]
        self.y = self.y[mask]
        self.ids = self.ids[mask]
        self.w = self.w[mask]

    def sort_by_y(self, ascending=True):
        """
        Sort the dataset based on the labels (y values).

        Args:
            ascending (bool, optional): If True, sorts in ascending order; if False, in descending order. Defaults to True.
        """
        sort_indices = np.argsort(self.y)

        if not ascending:
            sort_indices = sort_indices[::-1]

        self.X = self.X[sort_indices]
        self.y = self.y[sort_indices]
        self.ids = self.ids[sort_indices]
        self.w = self.w[sort_indices]

    def shuffle(self):
        """
        Shuffle the dataset randomly.

        This method applies a random permutation to the data, shuffling features, labels, identifiers,
        and weights in-place.
        """
        shuffle_indices = np.random.permutation(len(self.y))

        self.X = self.X[shuffle_indices]
        self.y = self.y[shuffle_indices]
        self.ids = self.ids[shuffle_indices]
        self.w = self.w[shuffle_indices]

    @staticmethod
    def merge_datasets(datasets):
        """
        Merge multiple Dataset objects into a single Dataset.

        Args:
            datasets (list of Dataset): List of Dataset objects to merge.

        Returns:
            Dataset: A new Dataset object that contains the concatenated data from all input datasets.
        """
        X, y, ids, w = [], [], [], []

        for dataset in datasets:
            X.append(dataset.X)
            y.append(dataset.y)
            ids.append(dataset.ids)
            w.append(dataset.w)

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        ids = np.concatenate(ids, axis=0)
        w = np.concatenate(w, axis=0)

        return Dataset(X, y, ids, w)

    @staticmethod
    def missing_points(original_dataset, model_dataset):
        """
        Identify and return data points that are present in the original dataset but missing in the model dataset.

        Args:
            original_dataset (Dataset): The original dataset.
            model_dataset (Dataset): The dataset representing the model's current data.

        Returns:
            Dataset: A Dataset object containing the data points that are missing in the model dataset.
        """
        points_in_model = np.isin(original_dataset.ids, model_dataset.ids, invert=True)
        dataset = original_dataset.get_points(points_in_model)

        return dataset

    def copy(self):
        """
        Create a deep copy of the dataset.

        Returns:
            Dataset: A deep copy of the current dataset.
        """
        import copy
        return copy.deepcopy(self)

    def remove_invalid_entries(self):
        """
        Remove data points where either the feature array (X) or the label array (y) contains NaN values.
        """
        invalid_indices_x = np.where(np.isnan(self.X).any(axis=1))[0]
        invalid_indices_y = np.where(np.isnan(self.y))[0]

        invalid_indices = np.unique(np.concatenate((invalid_indices_x, invalid_indices_y)))
        self.remove_points(invalid_indices)

    def keep_unlabel_entries_only(self):
        """
        Retain only data points where the label (y) is NaN.

        This method modifies the dataset in-place, keeping only the unlabeled data points.
        """
        unlabeled_data_indices = np.where(np.isnan(self.y))[0]

        self.X = self.X[unlabeled_data_indices]
        self.y = self.y[unlabeled_data_indices]
        self.ids = self.ids[unlabeled_data_indices]
        self.w = self.w[unlabeled_data_indices]

    @staticmethod
    def remove_mismatched_ids(*datasets):
        """
        Remove entries with non-matching IDs across multiple Dataset objects.

        Args:
            *datasets (Dataset): Variable number of Dataset objects.

        Returns:
            tuple: A tuple of Dataset objects with only the entries having common IDs across all datasets.
        """
        ids_sets = [set(dataset.ids) for dataset in datasets]
        common_ids = set.intersection(*ids_sets)
        common_ids = np.array(sorted(common_ids), dtype=datasets[0].ids.dtype)

        filtered_datasets = []
        for dataset in datasets:
            mask = np.isin(dataset.ids, common_ids)
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
        Check if all provided Dataset objects have the same IDs in the same order.

        Args:
            *datasets (Dataset): Variable number of Dataset objects.

        Returns:
            bool: True if all datasets have matching IDs in the same order, otherwise False.
        """
        if len(datasets) < 2:
            return True

        reference_ids = datasets[0].ids

        for dataset in datasets[1:]:
            if not np.array_equal(reference_ids, dataset.ids):
                return False

        return True

    def check_mismatches(self, *datasets):
        """
        Identify mismatched IDs across multiple Dataset objects.

        Args:
            *datasets (Dataset): Variable number of Dataset objects.

        Returns:
            dict: A dictionary where each key corresponds to a dataset (e.g., "dataset_0") and the value is a sorted
                  list of IDs that are missing in that dataset.
        """
        mismatches = {}
        dataset_ids_sets = [set(dataset.ids) for dataset in datasets]
        all_ids_set = set().union(*dataset_ids_sets)

        for i, dataset_ids in enumerate(dataset_ids_sets):
            mismatched_ids = all_ids_set - dataset_ids
            mismatches[f"dataset_{i}"] = sorted(mismatched_ids)

        return mismatches

    def get_top_or_bottom(self, n, highest=False):
        """
        Retrieve the top or bottom n data points based on the labels (y values).

        Args:
            n (int): Number of data points to retrieve.
            highest (bool, optional): If True, retrieves the data points with the highest y values;
                otherwise, retrieves the lowest. Defaults to False.

        Returns:
            Dataset: A new Dataset object containing the selected data points.
        """
        sorted_indices = np.argsort(self.y)
        if highest:
            selected_indices = sorted_indices[-n:]
        else:
            selected_indices = sorted_indices[:n]

        return Dataset(self.X[selected_indices], self.y[selected_indices], self.ids[selected_indices], self.w[selected_indices])

    def create_pairwise_dataset(self):
        """
        Create a pairwise dataset from the current dataset.

        The new dataset is constructed by pairing each data point with every other data point,
        concatenating their features, and including the difference between their features.

        Returns:
            Dataset: A new Dataset object representing the pairwise combinations of data points.
        """
        X1 = self.X
        X2 = self.X

        n1 = X1.shape[0]
        n2 = X2.shape[0]

        X1 = X1[:, np.newaxis, :].repeat(n2, axis=1)
        X2 = X2[np.newaxis, :, :].repeat(n1, axis=0)

        X = np.concatenate([X1, X2, X1 - X2], axis=2)
        X = X.reshape(n1 * n2, -1)

        y1 = self.y
        y2 = self.y

        y = (y1[:, np.newaxis] - y2[np.newaxis, :]).flatten()

        return Dataset(X, y)

    def split(self, test_size=0.2, shuffle=True):
        """
        Split the dataset into training and testing subsets.

        Args:
            test_size (float, optional): Proportion of the dataset to be used as the test set. Defaults to 0.2.
            shuffle (bool, optional): If True, shuffles the data before splitting. Defaults to True.

        Returns:
            tuple: A tuple of two Dataset objects, where the first is the training set and the second is the testing set.
        """
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, ids_train, ids_test, w_train, w_test = train_test_split(
            self.X, self.y, self.ids, self.w, test_size=test_size, random_state=None, shuffle=shuffle
        )
        return Dataset(X_train, y_train, ids_train, w_train), Dataset(X_test, y_test, ids_test, w_test)

    def remove_duplicates(self):
        """
        Remove duplicate entries from the dataset based on the 'ids' field.

        Only the first occurrence of each unique ID is retained. The dataset is modified in-place.
        """
        _, unique_indices = np.unique(self.ids, return_index=True)
        unique_indices = np.sort(unique_indices)

        self.X = self.X[unique_indices]
        self.y = self.y[unique_indices]
        self.ids = self.ids[unique_indices]
        self.w = self.w[unique_indices]

    def get_range(self, start, end):
        """
        Retrieve a subset of data points from the dataset based on a range of indices after sorting by labels.

        Args:
            start (int): The starting index of the range.
            end (int): The ending index of the range.

        Returns:
            Dataset: A new Dataset object containing the data points within the specified range.
        """
        temp_dataset = self.copy()
        temp_dataset.sort_by_y()
        indices = np.arange(start, end)
        return temp_dataset.get_points(indices)