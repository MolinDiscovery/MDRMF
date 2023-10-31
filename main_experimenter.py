from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/model_tests_morgan&rdkit2d.yaml")
exp.conduct_all_experiments()