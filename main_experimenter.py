from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/CDDD_descriptor_test.yaml")
exp.conduct_all_experiments()