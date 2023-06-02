# moleculeloader.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

class MoleculeLoader:
    """
    A class used to load and process molecular data.
    """
    def __init__(self, datafile: str, smi_col: str, scores_col: str) -> None:
        """
        Initializes the MoleculeLoader with a data file.

        Args:
            - datafile (str): The name of the data file to load.
            - smi_col (str): The name of the column containing SMILES strings in the `DataFrame`.
            - scores_col(str): The name of the column containing the docking scores in the `DataFrame`.
        """
        self.datafile = datafile
        self.smi_col = smi_col
        self.scores_col = scores_col
        self._df = None

    def _load_data(self) -> pd.DataFrame:
        """
        Loads data from the data file into a DataFrame.

        This method is called automatically when accessing the `df` property
        if the data has not been loaded yet.

        Returns:
            - pd.DataFrame: The loaded data.
        """
        self._df = pd.read_csv(self.datafile)

    def _process_data(self):
        """
        Processes the loaded DataFrame.
        """
        if 'molecules' not in self._df.columns:
            self._df['molecules'] = self._df[self.smi_col].apply(Chem.MolFromSmiles)

        #self._df.rename(columns={self.scores_col: "scores"}, inplace=True)
        #self._df.rename(columns={self.smi_col: "SMILES"}, inplace=True)

        self._df.sort_values(by=self.scores_col)

    @property
    def df(self):
        """
        Provides access to the loaded and processed DataFrame.

        If the data has not been loaded yet, it is loaded automatically
        the first time this property is accessed.

        Returns:
            - pd.DataFrame: The loaded DataFrame.
        """
        if self._df is None:
            self._load_data()
            self._process_data()
        return self._df

    def filter_by_mol_wt(self, MolWt: int):
        if self._df is None:
            raise ValueError("DataFrame is not loaded. Please load the data using .df on your object.")

        self._df = self._df[self._df['molecules'].apply(Descriptors.MolWt) < MolWt]

        return self

    def filter_by_num_atoms(self, max_atoms: int):
        if self._df is None:
            raise ValueError("DataFrame is not loaded. Please load the data using .df on your object.")
        
        self._df = self._df[self._df['molecules'].apply(lambda mol: mol.GetNumAtoms()) <= max_atoms]

        return self