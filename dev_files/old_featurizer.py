import pandas as pd
import numpy as np
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
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
            method (str): The featurization method to use. Supported methods are 'morgan' and 'topological'.
            **kwargs: Additional keyword arguments to pass to the featurization method.
        """
        print("Computing features...")

        if method == 'morgan':
            features_gen = tuple(self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(AllChem.GetMorganFingerprintAsBitVect(mol, **kwargs))))
            
        elif method == 'topological':
            features_gen = tuple(self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(FingerprintMols.FingerprintMol(mol, **kwargs))))
        elif method == 'MACCS':
            features_gen = tuple(self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(MACCSkeys.GenMACCSKeys(mol, **kwargs))))
        elif method == 'avalon':
            features_gen = tuple(self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(avalon.GetAvalonFP(mol, **kwargs))))
        elif method == 'rdk':
            features_gen = tuple(self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(Chem.RDKFingerprint(mol, **kwargs))))
        elif method == 'pharmacophore':
            pharm_factory = Gobbi_Pharm2D.factory
            features_gen = tuple(self.df[self.mol_col].apply(
                lambda mol: self._convert_to_np_array(Generate.Gen2DFingerprint(mol, pharm_factory))))
        # elif method == 'layered':
        #     features_gen = tuple(self.df[self.mol_col].apply(lambda mol: self._convert_to_np_array(AllChem.RDKFingerprint(mol, **kwargs))))
        elif method == 'mqn':
            mqn_descriptors = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors.MQNs])
            features_gen = tuple(self.df[self.mol_col].apply(lambda mol: mqn_descriptors.CalcDescriptors(mol)))
        elif method == 'rdkit2D':
            rdkit2D_descriptors = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors.descList])
            features_gen = tuple(self.df[self.mol_col].apply(lambda mol: rdkit2D_descriptors.CalcDescriptors(mol)))            
        else:
            raise ValueError(f"Unsupported featurization method: {method}")

        self.features = np.vstack(features_gen)

        print ("Feature compution completed.")
        return self.features

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


    # def _convert_to_np_array(self, bit_vect) -> np.ndarray:
    #     """
    #     Converts an RDKit explicit bit vector to a numpy array.

    #     Args:
    #         bit_vect: The bit vector to convert.

    #     Returns:
    #         np.ndarray: The converted numpy array.
    #     """
    #     np_array = np.zeros((1, bit_vect.GetNumBits()), dtype=np.int8)
    #     DataStructs.ConvertToNumpyArray(bit_vect, np_array)
    #     return np_array
    
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