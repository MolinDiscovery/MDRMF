#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/03-article/pairwise130k/01-RF-desc-pair-130k.yaml")
exp.conduct_all_experiments()