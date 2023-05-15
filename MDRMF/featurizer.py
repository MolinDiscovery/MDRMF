# featurizer.py

import pandas as pd
import numpy as np
import pickle
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs


class Featurizer:
    """
    A class to featurize molecules in a DataFrame.
    """
    def __init__(self, df: pd.DataFrame = None, mol_col: str = 'molecules') -> None:
        """
        Initializes the Featurizer with a DataFrame and the name of the column containing molecules.

        Args:
            df (pd.DataFrame): The DataFrame to featurize.
            mol_col (str): The name of the column containing molecules to featurize.
        """
        self.df = df
        self.mol_col = mol_col
        self.smi_col = 'SMILES'
        self.features = None

    def featurize(self, method: str, **kwargs) -> None:
        """
        Featurizes the molecules in the DataFrame using the specified method and stores the features separately.

        Args:
            method (str): The featurization method to use. Supported methods are 'morgan' and 'topological'.
            **kwargs: Additional keyword arguments to pass to the featurization method.
        """
        if method == 'morgan':
            self.df['features'] = self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(AllChem.GetMorganFingerprintAsBitVect(mol, **kwargs)))
        elif method == 'topological':
            self.df['features'] = self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(FingerprintMols.FingerprintMol(mol, **kwargs)))
        else:
            raise ValueError(f"Unsupported featurization method: {method}")
        
    def _convert_to_np_array(self, bit_vect) -> np.ndarray:
        """
        Converts an RDKit explicit bit vector to a numpy array.

        Args:
            bit_vect: The bit vector to convert.

        Returns:
            np.ndarray: The converted numpy array.
        """
        np_array = np.zeros((bit_vect.GetNumBits(),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bit_vect, np_array)
        return np_array


    def save(self, filename: str) -> None:
        """
        Saves the featurized DataFrame to a pickle file.

        Args:
            filename (str): The name of the file to save the featurized DataFrame to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.df, f)

    @classmethod
    def load(cls, filename: str) -> 'Featurizer':
        """
        Loads a featurized DataFrame from a pickle file.

        Args:
            filename (str): The name of the file to load the featurized DataFrame from.

        Returns:
            Featurizer: A Featurizer instance with the loaded DataFrame.
        """
        with open(filename, 'rb') as f:
            df = pickle.load(f)
        instance = cls(df=df)
        return instance
    
    def get_df(self):
        """
        Returns the featurized molecules for inspection.

        Returns:
            The featurized molecules.
        """
        if 'features' in self.df.columns:
            return self.df
        else:
            print("No features available. Please run the featurize method first.")

    def inspect_features_by_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """
        Inspects the features for a specific molecule based on its SMILES representation.

        Args:
            smiles (str): The SMILES string for the molecule to inspect.

        Returns:
            np.ndarray: The feature vector for the molecule, or None if the molecule is not found.
        """
        index = self.df[self.df[self.smi_col] == smiles].index
        if not index.empty:
            fingerprint = self.df['features'][index[0]]
            return fingerprint
        else:
            print(f"No molecule with SMILES {smiles} found in the DataFrame.")
            return None