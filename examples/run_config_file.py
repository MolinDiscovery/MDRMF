from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("01_simple_experiment.yaml")
exp.conduct_all_experiments()

# Alternatively using the command line interface
# > python -m MDRMF.experimenter 01_simple_experiment.yaml