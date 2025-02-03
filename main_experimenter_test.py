#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/04-article/DDS10/MLP_MQN_rdkit2d-10k_test.yaml")
exp.conduct_all_experiments()