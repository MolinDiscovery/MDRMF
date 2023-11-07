from multiprocessing import Pool
from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

def run_experiment(file_name):
    exp = Experimenter(file_name)
    exp.conduct_all_experiments()

if __name__ == "__main__":
    # List of your YAML files
    file_names = [
        "experiment_setups/FD_finalists_datasets1.yaml",
        "experiment_setups/FD_finalists_datasets2.yaml",
        "experiment_setups/FD_finalists_datasets3.yaml"
    ]

    # Create a pool of worker processes
    with Pool(processes=3) as pool:
        # Map the run_experiment function to the file names
        pool.map(run_experiment, file_names)