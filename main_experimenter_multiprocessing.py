import multiprocessing
from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

def run_experiment():
    exp = Experimenter("experiment_setups/FD_finalists_tests.yaml")
    exp.conduct_all_experiments()

if __name__ == '__main__':
    processes = []
    for _ in range(10):  # Number of processes to create
        p = multiprocessing.Process(target=run_experiment)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # Wait for all processes to finish