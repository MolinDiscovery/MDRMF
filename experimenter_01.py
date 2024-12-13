from MDRMF.experimenter import Experimenter
from MDRMF.dataset import Dataset
import psutil
import time
import os
import threading
import socket

def generate_bar(usage, length=10):
    """Generate a simple text-based bar for the given usage percentage."""
    bars = int((usage / 100.0) * length)
    return '█' * bars + '-' * (length - bars)

def format_cpu_usage(cpu_percents, cols=4):
    """Format CPU usage into multiple columns."""
    lines = []
    rows = (len(cpu_percents) + cols - 1) // cols  # Calculate the number of rows needed

    for row in range(rows):
        line = ""
        for col in range(cols):
            index = row + col * rows
            if index < len(cpu_percents):
                cpu_percent = cpu_percents[index]
                bar = generate_bar(cpu_percent)
                line += f"CPU {index:02}: {cpu_percent:5.1f}% {bar}    "
        lines.append(line)
    return "\n".join(lines)

def monitor_resources():

    current_dir = os.getcwd()

    hostname = socket.gethostname()

    log_file_path = os.path.join(current_dir, f"cpu_memory_usage_{hostname}.log")

    # Continuously update the log file with CPU and memory usage
    with open(log_file_path, "w") as f:
        while not stop_monitoring.is_set():
            f.seek(0) # find beginning of file

            cpu_percents = psutil.cpu_percent(interval=1, percpu=True)

            f.write("CPU Usage:\n")
            f.write(format_cpu_usage(cpu_percents, cols=4))

            memory_info = psutil.virtual_memory()
            f.write(f"\n\nMemory Usage: {memory_info.percent:.1f}% used ({memory_info.used / (1024**3):.2f} GB out of {memory_info.total / (1024**3):.2f} GB)\n")
            
            f.flush()
            f.truncate()

            time.sleep(1)

stop_monitoring = threading.Event()

monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

# >>>Insert your code below!! <<<
# Here I am using my active learning package (MDRMF), that reads a configuration file
# and the conduct_all_experiments() method executes it.
exp = Experimenter("experiment_setups/03-article/pairwise130k/01-RF-desc-pair-130k.yaml")
exp.conduct_all_experiments()

stop_monitoring.set()
monitor_thread.join()