from MDRMF.experimenter import Experimenter

exp = Experimenter("01_simple_experiment.yaml")
exp.conduct_all_experiments()

# Alternatively using the command line interface
# > python -m MDRMF.experimenter 01_simple_experiment.yaml