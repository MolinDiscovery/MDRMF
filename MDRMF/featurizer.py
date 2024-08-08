# featurizer.py

import pandas as pd
import numpy as np
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from rdkit.Avalon import pyAvalonTools as avalon
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.ML.Descriptors import MoleculeDescriptors
        
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
            method (str): The featurization method to use.
            **kwargs: Additional keyword arguments to pass to the featurization method.
        """
        print("Computing features...")

        # Dictionary mapping method names to corresponding functions
        method_funcs = {
            'morgan': AllChem.GetMorganFingerprintAsBitVect,
            'topological': FingerprintMols.FingerprintMol,
            'MACCS': MACCSkeys.GenMACCSKeys,
            'avalon': avalon.GetAvalonFP,
            'rdk': Chem.RDKFingerprint,
            'pharmacophore': self._generate_pharmacophore_fingerprint,
            'rdkit2D': self._generate_rdkit2D_fingerprint,
            'mqn': self._generate_mqn_fingerprint,
            # ... add other methods ...
        }

        if method not in method_funcs:
            raise ValueError(f"Unsupported featurization method: {method}")

        func = method_funcs[method]
        features_gen = []
        total_molecules = len(self.df)

        for idx, mol in enumerate(self.df[self.mol_col]):
            features_gen.append(self._convert_to_np_array(func(mol, **kwargs)))
            self._print_progress_bar(idx+1, total_molecules)

        self.features = np.vstack(tuple(features_gen))
        print ("\nFeature computation completed.")
        return self.features    

    def _generate_pharmacophore_fingerprint(self, mol, **kwargs):
        pharm_factory = Gobbi_Pharm2D.factory
        return Generate.Gen2DFingerprint(mol, pharm_factory)
    
    
    def _generate_rdkit2D_fingerprint(self, mol, **kwargs):
        rdkit2D_descriptors = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors.descList])
        descriptors = np.array(rdkit2D_descriptors.CalcDescriptors(mol))
        # Replace NaN values with the mean of the respective descriptor
        mean_values = np.nanmean(descriptors)
        descriptors = np.where(np.isnan(descriptors), mean_values, descriptors)
        return descriptors

    
    def _generate_mqn_fingerprint(self, mol, **kwargs):
        return rdMolDescriptors.MQNs_(mol)

    def _print_progress_bar(self, iteration, total, bar_length=50):
        """
        Print the progress bar.

        Args:
            iteration (int): current iteration.
            total (int): total iterations.
            bar_length (int): length of the progress bar.
        """
        progress = (iteration / total)
        arrow = '-' * int(round(progress * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        # Format the progress bar string to include molecule count and total count
        print(f"\rProgress: [{arrow + spaces}] {int(progress * 100)}% ({iteration}/{total})", end='')


    def _convert_to_np_array(self, data) -> np.ndarray:
        """
        Converts an RDKit data (bit vector or tuple) to a numpy array.

        Args:
            data: The data to convert.

        Returns:
            np.ndarray: The converted numpy array.
        """
        if isinstance(data, DataStructs.ExplicitBitVect):
            np_array = np.zeros((1, data.GetNumBits()), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(data, np_array)
        else:  # Assume data is a tuple of descriptor values
            np_array = np.array(data).reshape(1, -1)
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
