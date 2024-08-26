#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/article 30n/experiment_setups/03-article/06-MLP-desc-pair-10k.yaml")
exp.conduct_all_experiments()