from multiprocessing import Pool
from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

def run_experiment(file_name):
    exp = Experimenter(file_name)
    exp.conduct_all_experiments()

if __name__ == "__main__":
    # List of your YAML files
    file_names = [
        "experiment_setups/article/(article) 13 noise nudged test 10k.yaml",
        "experiment_setups/article/(article) 14 noise nudged test 140k.yaml",
    ]

    # # Create a pool of worker processes
    # with Pool(processes=2) as pool:
    #     # Map the run_experiment function to the file names
    #     pool.map(run_experiment, file_names)

    for file_name in file_names:
        run_experiment(file_name)