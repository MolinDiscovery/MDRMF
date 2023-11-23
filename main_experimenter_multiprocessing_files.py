from multiprocessing import Pool
from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

def run_experiment(file_name):
    exp = Experimenter(file_name)
    exp.conduct_all_experiments()

if __name__ == "__main__":
    # List of your YAML files
    file_names = [
        "experiment_setups/FD_finalists_tests_seed1.yaml",
        "experiment_setups/FD_finalists_tests_seed2.yaml",
        "experiment_setups/FD_finalists_tests_seed3.yaml",
        "experiment_setups/FD_finalists_tests_seed4.yaml",
        "experiment_setups/FD_finalists_tests_seed5.yaml",
        "experiment_setups/FD_finalists_tests_seed6.yaml",
        "experiment_setups/FD_finalists_tests_seed7.yaml",
        "experiment_setups/FD_finalists_tests_seed8.yaml",
        "experiment_setups/FD_finalists_tests_seed9.yaml",
        "experiment_setups/FD_finalists_tests_seed10.yaml",
    ]

    # Create a pool of worker processes
    with Pool(processes=3) as pool:
        # Map the run_experiment function to the file names
        pool.map(run_experiment, file_names)