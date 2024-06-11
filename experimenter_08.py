#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/article 30n/12-5T nudging tests 130k.yaml")
exp.conduct_all_experiments()