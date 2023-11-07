from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/FD_finalists_tests.yaml")
exp.conduct_all_experiments()