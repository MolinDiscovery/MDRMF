# MDRMF Examples

Here is given some example configuration files and scripts to help you get started with MDRMF.

## Configuration Examples

### 1. Simple Experiment (`01_simple_experiment.yaml`)
A basic example showing how to run a single experiment:
- Uses Random Forest model
- Reads data directly from CSV file
- Uses greedy acquisition strategy
- Evaluates results with top-k metrics

### 2. Creating Featurized Dataset (`02_create_featurized_dataset.yaml`) 
Demonstrates how to create and save a featurized dataset:
- Creates a dataset named "10K_MACCS"
- Processes molecules from CSV using MACCS fingerprints
- Allows dataset shuffling

### 3. Using Pre-featurized Dataset (`03_using_prefeaturized_dataset.yaml`)
Shows how to run experiments using a pre-featurized dataset:
- Uses MLP (Multi-Layer Perceptron) model
- Loads pre-featurized dataset from pickle file
- Uses Expected Improvement (EI) acquisition method

### 4. Running Multiple Experiments (`04_running_multiple_experiments.yaml`)
Demonstrates how to run multiple experiments in a single configuration:
- Compares different acquisition methods (EI vs greedy)
- Both experiments use the same dataset and model type
- Helpful for comparative analysis

## Interactive Examples

### 5. Inspecting Datasets (`05_inspect_dataset.ipynb`)
A Jupyter notebook demonstrating how to:
- Load and inspect datasets
- View feature vectors and scores
- Shuffle and sort datasets
- Save modified datasets

## How to Run Examples

You can run the YAML configuration files using either the Python API or command line interface:

### Using Python API:
```python
from MDRMF.experimenter import Experimenter
exp = Experimenter("01_simple_experiment.yaml")
exp.conduct_all_experiments()
```

### Using Command Line:
```bash
python -m MDRMF.experimenter 01_simple_experiment.yaml
```

For the Jupyter notebook, open it in a notebook environment:
```bash
jupyter notebook 05_inspect_dataset.ipynb
```

