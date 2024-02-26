#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/spring_experiments/acq_func_tests.yaml")
exp.conduct_all_experiments()