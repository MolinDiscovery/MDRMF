from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

exp = Experimenter("experiment_setups/all_descriptors_mixed_seeded_prefeaturized.yaml")
exp.conduct_all_experiments()