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
    # Get the current directory
    current_dir = os.getcwd()

    # Get the hostname to ensure the log file is unique to this node
    hostname = socket.gethostname()

    # Set the log file path in the current directory with the hostname included
    log_file_path = os.path.join(current_dir, f"cpu_memory_usage_{hostname}.log")

    # Continuously update the log file with current CPU and memory usage
    with open(log_file_path, "w") as f:
        while not stop_monitoring.is_set():
            # Seek to the beginning of the file to overwrite it
            f.seek(0)

            cpu_percents = psutil.cpu_percent(interval=1, percpu=True)

            # Format and display CPU usage in multiple columns
            f.write("CPU Usage:\n")
            f.write(format_cpu_usage(cpu_percents, cols=4))

            # Display memory usage
            memory_info = psutil.virtual_memory()
            f.write(f"\n\nMemory Usage: {memory_info.percent:.1f}% used ({memory_info.used / (1024**3):.2f} GB out of {memory_info.total / (1024**3):.2f} GB)\n")
            
            # Flush the file to ensure content is written out and truncate leftover content
            f.flush()
            f.truncate()

            time.sleep(1)

# Create an event to signal when to stop monitoring
stop_monitoring = threading.Event()

# Start the monitoring thread
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

# Run the experiments
exp = Experimenter("experiment_setups/04-article/enrichmentOuter130k-pair/01_enrichment130k_MQN_RF_Greedy-pair-outer.yaml")
exp.conduct_all_experiments()

# Signal the monitoring thread to stop and wait for it to finish
stop_monitoring.set()
monitor_thread.join()