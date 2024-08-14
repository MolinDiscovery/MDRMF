#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("/home/jmni/MDRMF/experiments/pairwise_test_130k.yaml")
exp.conduct_all_experiments()