#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/article 30n/17 pairwise test 10k.yaml")
exp.conduct_all_experiments()