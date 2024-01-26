#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/pairwise_vs_descriptor.yaml")
exp.conduct_all_experiments()