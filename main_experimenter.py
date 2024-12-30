#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/04-article/DDS10/noise10k_LGBM_rdkit2D.yaml")
exp.conduct_all_experiments()