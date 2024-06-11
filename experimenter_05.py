#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/article 30n/15 feature importance tests 10k.yaml")
exp.conduct_all_experiments()