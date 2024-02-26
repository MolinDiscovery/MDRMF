from multiprocessing import Pool
from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

def run_experiment(file_name):
    exp = Experimenter(file_name)
    exp.conduct_all_experiments()

if __name__ == "__main__":
    # List of your YAML files
    file_names = [
        "experiment_setups/one_percent.yaml",
        "experiment_setups/one_percent_nudged.yaml",
    ]

    # Create a pool of worker processes
    with Pool(processes=2) as pool:
        # Map the run_experiment function to the file names
        pool.map(run_experiment, file_names)