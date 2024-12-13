from multiprocessing import Pool
from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

def run_experiment(file_name):
    exp = Experimenter(file_name)
    exp.conduct_all_experiments()

if __name__ == "__main__":
    # List of your YAML files
    file_names = [
        # "experiment_setups/03-article/130k/01-RF-desc-130k.yaml",
        # "experiment_setups/03-article/130k/02-DT-desc-130k.yaml",
        # "experiment_setups/03-article/130k/03-SVR-desc-130k.yaml",
        "experiment_setups/03-article/130k/04-LGBM-desc-130k.yaml",
        # "experiment_setups/03-article/130k/05-KNN-desc-130k.yaml",
        "experiment_setups/03-article/130k/06-MLP-desc-130k.yaml"
    ]

    # # Create a pool of worker processes
    # with Pool(processes=2) as pool:
    #     # Map the run_experiment function to the file names
    #     pool.map(run_experiment, file_names)

    for file_name in file_names:
        run_experiment(file_name)