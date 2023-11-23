import multiprocessing
from multiprocessing import Pool
import os
import sys
import io
import time
from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset

def format_time(seconds):

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        time_str = "{} hour(s), {} minute(s), {} second(s)".format(int(hours), int(minutes), int(seconds))
    elif minutes:
        time_str = "{} minute(s), {} second(s)".format(int(minutes), int(seconds))
    else:
        time_str = "{} second(s)".format(int(seconds))

    return time_str

def run_experiment(start_time, elapsed_time_shared):
    sys.stdout = io.StringIO()
    
    exp = Experimenter("experiment_setups/FD_finalists_tests.yaml")
    exp.conduct_all_experiments()
    
    sys.stdout = sys.__stdout__

    # Update the shared elapsed time value
    elapsed_time_shared.value = time.time() - start_time

def print_elapsed_time(start_time, elapsed_time_shared):
    while True:
        elapsed_time = time.time() - start_time
        sys.stdout.write("\rTime elapsed: " + format_time(elapsed_time))
        sys.stdout.flush()
        time.sleep(1)  # Update every second

        # If processes are done, break the loop
        if elapsed_time_shared.value != -1:
            sys.stdout.write("\rFinal time elapsed: " + format_time(elapsed_time_shared.value) + "\n")
            sys.stdout.flush()
            break

if __name__ == '__main__':
    start_time = time.time()
    elapsed_time_shared = multiprocessing.Value('d', -1.0)

    # Start the time printing process
    printer_process = multiprocessing.Process(target=print_elapsed_time, args=(start_time, elapsed_time_shared))
    printer_process.start()

    # Create a pool of 4 worker processes
    with Pool(4) as pool:
        # Start 10 calls to run_experiment using the pool
        # The pool will only run 4 of these at a time
        pool.starmap(run_experiment, [(start_time, elapsed_time_shared) for _ in range(10)])

    # Since starmap is blocking, we don't need to manually join the processes

    # Update the shared value to indicate completion
    elapsed_time_shared.value = time.time() - start_time
    printer_process.join()  # Wait for the printer process to finish