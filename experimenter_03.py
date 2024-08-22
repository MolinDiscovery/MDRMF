#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/03-article/01-desc-tests-10k.yaml")
exp.conduct_all_experiments()