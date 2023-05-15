# featurizer.py

import pandas as pd
import numpy as np
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
            # self.df['features'] = self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(AllChem.GetMorganFingerprintAsBitVect(mol, **kwargs)))
            features_gen = tuple(self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(AllChem.GetMorganFingerprintAsBitVect(mol, **kwargs))))
        elif method == 'topological':
            features_gen = tuple(self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(FingerprintMols.FingerprintMol(mol, **kwargs))))
            # self.df['features'] = self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(FingerprintMols.FingerprintMol(mol, **kwargs)))
        else:
            raise ValueError(f"Unsupported featurization method: {method}")

        self.features = np.vstack(features_gen)

        return self.features

    def _convert_to_np_array(self, bit_vect) -> np.ndarray:
        """
        Converts an RDKit explicit bit vector to a numpy array.

        Args:
            bit_vect: The bit vector to convert.

        Returns:
            np.ndarray: The converted numpy array.
        """
        np_array = np.zeros((1, bit_vect.GetNumBits()), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bit_vect, np_array)
        return np_array
    
    def get_df(self):
        """
        Returns:
            The DataFrame
        """
        return self.df

    def get_features(self):
        """
        Returns the 2D numpy array of featurized molecules.

        Returns:
            np.ndarray: The featurized molecules.
        """
        if self.features is not None:
            return self.features
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