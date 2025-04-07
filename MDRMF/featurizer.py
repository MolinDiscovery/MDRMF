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
    A class to featurize molecules contained in a pandas DataFrame.

    This class provides various methods to compute molecular features
    (fingerprints and descriptors) that can be used for machine learning,
    cheminformatics, or data analysis tasks.

    Available featurizers:
        - "morgan": Morgan circular fingerprints.
        - "topological": Topological fingerprints.
        - "MACCS": MACCS keys fingerprints.
        - "avalon": Avalon fingerprints.
        - "rdk": RDKit default fingerprints.
        - "pharmacophore": Pharmacophore fingerprints based on Gobbi's approach.
        - "rdkit2D": 2D descriptors computed by RDKit.
        - "mqn": Molecular Quantum Numbers (MQN).

    Usage:
        1. Initialize the Featurizer with a DataFrame containing the rdkit molecule objects.
        2. Call the featurize method with the desired method name and any additional kwargs.
        3. Retrieve the computed features using get_features(), or inspect individual features using inspect_features_by_smiles().

    Attributes:
        df (pd.DataFrame): DataFrame holding the molecules.
        mol_col (str): The column name in df containing molecule objects.
        smi_col (str): The column name for SMILES representation.
        features (np.ndarray): The 2D array of computed molecular features.
    """

    def __init__(self, df: pd.DataFrame = None, mol_col: str = 'molecules') -> None:
        """
        Initialize the Featurizer.

        Args:
            df (pd.DataFrame, optional): DataFrame containing molecules.
            mol_col (str, optional): Column name in df with molecule objects.
        """
        self.df = df
        self.mol_col = mol_col
        self.smi_col = 'SMILES'
        self.features = None

    def featurize(self, method: str, **kwargs) -> np.ndarray:
        """
        Featurize molecules in the DataFrame using a specified method.

        This method computes the molecular features using one of the supported
        featurization algorithms and stores them as a 2D numpy array.

        Args:
            method (str): The featurization method to use. Supported values are:
                "morgan", "topological", "MACCS", "avalon", "rdk",
                "pharmacophore", "rdkit2D", and "mqn".
            **kwargs: Additional keyword arguments to be passed to the
                selected featurization function.

        Returns:
            np.ndarray: A 2D numpy array where each row corresponds to the
                computed features of a molecule.

        Raises:
            ValueError: If the provided featurization method is not supported.
        """
        print("Computing features...")

        # Dictionary mapping method names to corresponding functions
        method_funcs = {
            'morgan': self._generate_morgan_fp,
            'topological': FingerprintMols.FingerprintMol,
            'MACCS': MACCSkeys.GenMACCSKeys,
            'avalon': avalon.GetAvalonFP,
            'rdk': Chem.RDKFingerprint,
            'pharmacophore': self._generate_pharmacophore_fingerprint,
            'rdkit2D': self._generate_rdkit2D_fingerprint,
            'mqn': self._generate_mqn_fingerprint,
            # ... add other methods if needed ...
        }

        if method not in method_funcs:
            raise ValueError(f"Unsupported featurization method: {method}")

        func = method_funcs[method]
        features_gen = []
        total_molecules = len(self.df)

        for idx, mol in enumerate(self.df[self.mol_col]):
            features_gen.append(self._convert_to_np_array(func(mol, **kwargs)))
            self._print_progress_bar(idx + 1, total_molecules)

        self.features = np.vstack(tuple(features_gen))
        print("\nFeature computation completed.")
        return self.features

    def _generate_pharmacophore_fingerprint(self, mol, **kwargs):
        """
        Generate a pharmacophore fingerprint for the molecule.

        Args:
            mol: An RDKit molecule object.
            **kwargs: Additional keyword arguments.

        Returns:
            The pharmacophore fingerprint.
        """
        pharm_factory = Gobbi_Pharm2D.factory
        return Generate.Gen2DFingerprint(mol, pharm_factory)
    
    def _generate_rdkit2D_fingerprint(self, mol, **kwargs):
        """
        Generate 2D descriptors fingerprint using RDKit.

        This method computes 2D descriptors for a molecule, handling NaN values
        by replacing them with the mean across descriptors.

        Args:
            mol: An RDKit molecule object.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: 2D descriptor values as a numpy array.
        """
        rdkit2D_descriptors = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors.descList])
        descriptors = np.array(rdkit2D_descriptors.CalcDescriptors(mol))
        mean_values = np.nanmean(descriptors)
        descriptors = np.where(np.isnan(descriptors), mean_values, descriptors)
        return descriptors

    def _generate_mqn_fingerprint(self, mol, **kwargs):
        """
        Generate Molecular Quantum Numbers (MQN) fingerprint.

        Args:
            mol: An RDKit molecule object.
            **kwargs: Additional keyword arguments.

        Returns:
            The MQN fingerprint.
        """
        return rdMolDescriptors.MQNs_(mol)

    def _generate_morgan_fp(self, mol, **kwargs):
        """
        Generate Morgan fingerprint using the new MorganGenerator.
        Caches the generator instance to avoid re-creating it for every molecule.

        Args:
            mol: An RDKit molecule object.
            **kwargs: Optional keyword parameters.
                radius (int): Radius for Morgan fingerprint (default: 2).
                nBits (int): Number of bits for the fingerprint (default: 2048).

        Returns:
            RDKit fingerprint object from the MorganGenerator.
        """
        radius = kwargs.get('radius', 2)
        nBits = kwargs.get('nBits', 2048)
        if not hasattr(self, '_morgan_generator'):
            self._morgan_generator = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
        return self._morgan_generator.GetFingerprint(mol)

    def _print_progress_bar(self, iteration, total, bar_length=50):
        """
        Print a progress bar in the terminal.

        Args:
            iteration (int): Current iteration number.
            total (int): Total number of iterations.
            bar_length (int, optional): The length of the progress bar. Defaults to 50.
        """
        progress = iteration / total
        arrow = '-' * int(round(progress * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        print(f"\rProgress: [{arrow + spaces}] {int(progress * 100)}% ({iteration}/{total})", end='')

    def _convert_to_np_array(self, data) -> np.ndarray:
        """
        Convert RDKit output (bit vector or tuple) to a numpy array.

        Args:
            data: RDKit fingerprint or descriptor data.

        Returns:
            np.ndarray: The converted numpy array.
        """
        if isinstance(data, DataStructs.ExplicitBitVect):
            np_array = np.zeros((1, data.GetNumBits()), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(data, np_array)
        else:
            np_array = np.array(data).reshape(1, -1)
        return np_array

    def get_df(self) -> pd.DataFrame:
        """
        Get the DataFrame containing molecule data.

        Returns:
            pd.DataFrame: The DataFrame used for featurization.
        """
        return self.df

    def get_features(self) -> Optional[np.ndarray]:
        """
        Retrieve the featurized molecules as a 2D numpy array.

        Returns:
            np.ndarray: The array of computed features if available; otherwise, None.
        """
        if self.features is not None:
            return self.features
        else:
            print("No features available. Please run the featurize method first.")

    def inspect_features_by_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """
        Inspect the computed features for a molecule given its SMILES.

        Args:
            smiles (str): SMILES string of the molecule to inspect.

        Returns:
            np.ndarray: The feature vector for the molecule, or None if not found.
        """
        index = self.df[self.df[self.smi_col] == smiles].index
        if not index.empty:
            fingerprint = self.df['features'][index[0]]
            return fingerprint
        else:
            print(f"No molecule with SMILES {smiles} found in the DataFrame.")
            return None
