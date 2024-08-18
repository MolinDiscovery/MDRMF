#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/dev_tests/retrieve_init.yaml")
exp.conduct_all_experiments()