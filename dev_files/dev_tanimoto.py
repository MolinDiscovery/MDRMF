import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect 

def similarity_search(hits, smiles, n: int = 1, radius: int = 2, nBits: int = 1024, **kwargs):
    """ 1. Compute the similarity of all screen smiles to all hit smiles
        2. take the n screen smiles with the highest similarity to any hit """

    fp_hits = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=radius, nBits=nBits) for smi in hits]
    fp_smiles = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=radius, nBits=nBits) for smi in smiles]

    m = np.zeros([len(hits), len(smiles)], dtype=np.float16)
    for i in range(len(hits)):
        m[i] = BulkTanimotoSimilarity(fp_hits[i], fp_smiles)

    print(m)
    # get the n highest similarity smiles to any hit
    picks_idx = np.argsort(np.max(m, axis=0))[::-1][:n]

    # Correctly index into smiles using picks_idx
    selected_smiles = np.array(smiles)[picks_idx]

    return selected_smiles

# Example usage
# hits = ['CCO', 'CCN', 'CCO']
# smiles = ['NCCO', 'CCNN', 'CCO', 'CCCO', 'CCN', 'CCCN']
hits = ['CC', 'CO']
smiles = ['CC', 'COC', 'CCO', 'COO']
result = similarity_search(hits, smiles, n=2, radius=2, nBits=1024)
print(result)