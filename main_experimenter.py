#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/04-article/DDS10/acquisition10k_MLP_MQN.yaml")
exp.conduct_all_experiments()