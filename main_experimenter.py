#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/pairwise_vs_descriptor.yaml")
#exp = Experimenter("experiment_setups/pairwise_vs_descriptor.yaml", pre_run=True)
exp.conduct_all_experiments()