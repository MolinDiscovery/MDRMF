#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/dev_tests/test3.yaml")
exp.conduct_all_experiments()