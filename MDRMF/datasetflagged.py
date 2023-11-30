# datasetflagged.py

import pandas as pd
import numpy as np
from MDRMF import Dataset

class FlaggedDataset(Dataset):
    def __init__(self, X, y, ids=None, w=None) -> None:
        super().__init__(X, y, ids, w)
        self.is_labeled = ~pd.isna(y)

    def update_labels(self, new_labels):
        """
        Update the dataset with new labels.

        Parameters:
        new_labels (dict): A dictionary where keys are the ids and values are the new labels.
        """
        for id, label in new_labels.items():
            index = np.where(self.ids == id)[0]
            if index.size > 0:
                self.y[index] = label
                self.is_labeled = True