import multiprocessing
import os
import sys
import io
from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

def run_experiment():
    # Redirect standard output to a dummy StringIO object
    sys.stdout = io.StringIO()
    
    try:
        exp = Experimenter("experiment_setups/FD_finalists_tests.yaml")
        exp.conduct_all_experiments()
    finally:
        # Ensure that we reset sys.stdout to the original state after the function is done
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    processes = []
    for _ in range(10):  # Number of processes to create
        p = multiprocessing.Process(target=run_experiment, name=f'Experiment {_}')
        p.start()
        print(f"Started process: {p}")
        processes.append(p)

    for p in processes:
        p.join()  # Wait for all processes to finish
        print(f"Finished process: {p}")