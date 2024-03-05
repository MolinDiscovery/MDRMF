#main_experimenter.py

from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/structural_biology_presentation_140k.yaml")
exp.conduct_all_experiments()