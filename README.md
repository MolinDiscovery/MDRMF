# Table of Contents

- [Table of Contents](#table-of-contents)
- [Why is it called MDRMF?](#why-is-it-called-mdrmf)
- [What does MDRMF do?](#what-does-mdrmf-do)
- [Installation](#installation)
- [How to use MDRMF](#how-to-use-mdrmf)
  - [Testing your setup (retrospective study)](#testing-your-setup-retrospective-study)
    - [Execution](#execution)
    - [data](#data)
    - [featurizer](#featurizer)
    - [model](#model)
    - [metrics](#metrics)
    - [Configuration documentation](#configuration-documentation)
      - [Create multiple experiments](#create-multiple-experiments)
      - [Create a pre-featurized dataset](#create-a-pre-featurized-dataset)
      - [Use a prefeaturized dataset](#use-a-prefeaturized-dataset)
      - [Using a prefeaturized CSV file.](#using-a-prefeaturized-csv-file)
      - [Use SMILES as seeds for every experiment.](#use-smiles-as-seeds-for-every-experiment)
      - [Generate random seeds, but use the same random seeds for every experiment.](#generate-random-seeds-but-use-the-same-random-seeds-for-every-experiment)
      - [Acquisition functions](#acquisition-functions)
      - [Adding noise to the data](#adding-noise-to-the-data)
      - [Data enrichment](#data-enrichment)
      - [Feature importance](#feature-importance)
- [Using experimental data (prospective study)](#using-experimental-data-prospective-study)

# Why is it called MDRMF?
MDRMF is a Python package that was developed as part of a project to discover inhibitors of ABC transporters, which drive multiple-drug resistance toward various chemotherapeutics. The “machine fishing” part refers to the idea that active learning can be seen as fishing for drug candidates in an ocean of molecules.

# What does MDRMF do?
MDRMF is a platform that helps find candidate drugs for a particular disease target. The software has two modes:

1) A **retrospective mode** for testing and optimizing the active learning workflow.
2) A **prospective mode** for usage in experimental settings.

**Retrospective mode:** This is for testing and optimization. You have a dataset of SMILES that is fully labeled with a score (e.g., a docking score). The software can then evaluate how many hits it can obtain with the chosen settings.

**Prospective mode:** The software is designed to be used on experimental data. You have a list of SMILES in your dataset. You select X number of molecules and experimentally test them to get labels. These measured labels are then fed back into the software to train a model. The software will then propose X molecules to test in the next round.

# Installation
```bash
pip install MDRMF
```

Ensure the required dependencies are installed. Preferentially, create a conda environment from `environment.yaml`:

```bash
conda env create -f environment.yaml
```
This will also install MDRMF itself.

# How to use MDRMF
MDRMF works by reading YAML configuration files that define the experiments you want to run. When you conduct an experiment, it will create a directory matching the configuration file’s name. In that directory, it will store results for each experiment along with various artifacts such as training datasets, graphs, settings, and more.

## Testing your setup (retrospective study)
Below is an example of a simple configuration file for a retrospective experiment:

```yaml
- Experiment:
    name: retrospective_docking_experiment

    data:
      datafile: docking_data.csv
      SMILES_col: SMILES
      scores_col: docking_score
      ids_col: SMILES

    featurizer:
      name: morgan

    model:
        name: RF
        iterations: 5
        initial_sample_size: 10
        acquisition_size: 20
        acquisition_method: greedy

    metrics:
        names: [top-k-acquired]
        k: [100]
```

This configuration specifies one experiment named `retrospective_docking_experiment`.

There are two ways to execute this file: via the CLI or within a Python script.

### Execution
**CLI**:
```bash
python -m MDRMF.experimenter config-file.yaml
```

**Python**:
```python
exp = Experimenter("config-file.yaml")
exp.conduct_all_experiments()
```

### data
In the above experiment, a `.csv` file is read, specifying two required columns: `SMILES_col` and `scores_col`. An optional `ids_col` is set to the SMILES column in the `.csv` file (if left unspecified, a sequential list of numbers will be generated for IDs).

### featurizer
The `featurizer` section tells MDRMF how to describe the molecules. The currently supported featurizers are:
```python
morgan, topological, MACCS, avalon, rdk, pharmacophore, rdkit2D, mqn
```
These are all implementations from RDKit, and you can pass arguments directly to them. For instance, if you want Morgan fingerprints with a specific bit vector length, you can specify:

```yaml
featurizer:
  name: morgan
  nBits: 2048
```

### model
This section defines the machine learning model and the active learning parameters. In our example, we specify a random forest model (`RF`) to be initialized with 10 random molecules. At each iteration, 20 new molecules are acquired, for 5 iterations total.

All the models except **LightGBM** come directly from the scikit-learn package. As with the featurizer, you can pass arguments directly to the underlying model. For example, if you have a multicore CPU, you can pass `n_jobs`:

```yaml
model:
  name: RF
  iterations: 5
  initial_sample_size: 10
  acquisition_size: 20
  acquisition_method: greedy
  n_jobs: 14  # define number of cores
```

Currently supported models are:
```python
RF (Random Forest), MLP (Multi-layer perceptron), KNN (K-nearest neighbors),
LGBM (LightGBM), DT (DecisionTree), SVR (Support Vector Regressor)
```
⚠️ **Note**: MDRMF only uses regression models. Classification is not supported.

### metrics
Metrics define how to evaluate the active learning experiment. In the demo configuration, we track `top-k-acquired`, i.e., how many of the highest 100 scored molecules were found in the training set at each iteration. MDRMF’s evaluators include:

```yaml
[top-k-acquired, R2_model, R2_k, top-k]
```
You can do multiple evaluations by supplying a longer list, for example, `k: [100, 1000]`.

- **top-k** returns how many of the top-k molecules (by true score) the model also predicts to be in the top-k. If the model’s predictions align exactly with the top-k, it will have a higher metric value.  
- **R2_model, R2_k** measure how well the model is performing (e.g., the R^2 on subsets of data).

**Note**: Using `R2_model`, `R2_k`, and `top-k` involves extra predictions during active learning iterations. If you’re conducting pairwise (PADRE) experiments, that can be resource-intensive, so be mindful if you choose to use these.

### Configuration documentation

#### Create multiple experiments
```yaml
- Experiment:
    name: exp1
    # setup ...

- Experiment:
    name: exp2
    # setup ...
```

#### Create a pre-featurized dataset
```yaml
- create_dataset:
    name: dataset_morgan

    data:
      datafile: 10K.csv
      SMILES_col: SMILES
      scores_col: docking_scores
      ids_col: SMILES

    featurizer:
      name: morgan
      nBits: 1024
      radius: 2
```
This will create a `.pkl` (pickle) file that can be used for experiments.

#### Use a prefeaturized dataset
```yaml
- Experiment:
    name: exp-using-dataset

    dataset: path/to/dataset.pkl

    model:
      # ...
    metrics:
      # ...
```
Because the dataset is already featurized, you do not need to specify `featurizer` when using a pre-featurized dataset.

#### Using a prefeaturized CSV file.
MDRMF can also work with CSV files containing pre-computed features. This lets you generate your own features and import them directly into MDRMF.

```yaml
- Experiment:
    name: retrospective_docking_experiment

    data:
      datafile: docking_data.csv
      vector_col: features
      scores_col: docking_score
      ids_col: SMILES

    # ...
```
You can also create a dataset directly using the `create_dataset` keyword instead of `Experiment`.  
(See: [Create a pre-featurized dataset](#create-a-pre-featurized-dataset))

#### Use SMILES as seeds for every experiment.
You can specify an initial set of SMILES to be used in every experiment. This overrides the `initial_sample_size` argument in the model setup. The example below shows two replicates for every experiment, each with its own list of initial SMILES:

```yaml
- unique_initial_sample:
    seeds: [
      [
        'O=C(Nc1ccc(Nc2ncccn2)cc1)c1cn[nH]c1-c1ccco1',
        'NC(=O)c1ccc(C(=O)N2CCC[C@H](Cn3ccnn3)C2)nc1',
        'COc1ccnc(NC[C@]23C[C@](NC(=O)[C@@H]4C[C@@H]4C)(C2)C(C)(C)O3)n1',
        'Cc1csc(N2CCN(C(=O)c3ccc(C(=O)NC4CC4)cc3)C[C@H]2C)n1',
        'CN1C(=O)CCc2cc(NC(=O)NC[C@@H](O)c3ccccc3)ccc21',
      ],
      [
        'O=C([O-])c1cccc(CS(=O)(=O)N2CC[C@H](O)C2)c1',
        'O=C(CCc1cccc(Br)c1)N[C@H]1C[C@H](Cn2ccnc2)C[C@@H]1O',
        'Cc1ccccc1CNc1cc(C(N)=O)ccc1Cl',
        'COc1ccc(OC)c([C@@H]2CCCN2C(=O)c2ccnc(OC)n2)c1',
        'C=CCN(CC(=O)[O-])S(=O)(=O)c1ccc(OC)c(Cl)c1',
      ]
      # ...
    ]

- Experiment:
  # setup ...

- Experiment:
  # setup ...
```

#### Generate random seeds, but use the same random seeds for every experiment.
This also overrides the `initial_sample_size` in the model setup:
```yaml
- unique_initial_sample:
    sample_size: 10

- Experiment:
  # setup ...

- Experiment:
  # setup ...
```

#### Acquisition functions
Active learning, like other low-data machine learning scenarios, involves balancing exploration vs. exploitation. MDRMF implements the following seven acquisition functions:

```python
'greedy', 'MU' (most uncertainty), 'LCB' (lower confidence bound), 
'EI' (expected improvement), 'TS' (Thompson sampling), 'tanimoto', 'random'.
```
Pick the function you want:
```yaml
- Experiment:
    ...
    model:
      name: RF
      iterations: 5
      initial_sample_size: 10
      acquisition_size: 20
      acquisition_method: greedy  # or MU, LCB, EI, TS, tanimoto, random
    ...
```
**Note**: Only `RF`, `KNN`, and `LGBM` can use `MU`, `LCB`, `EI`, or `TS` because these require an uncertainty estimate.

#### Adding noise to the data
To simulate a prospective study (e.g., an _in vitro_ study) and introduce measurement variability, you can add noise to the labels:

```yaml
- Experiment:
    ...
    model:
        name: RF
        iterations: 5
        initial_sample_size: 10
        acquisition_size: 20
        add_noise: 1
    ...
```
In this example, at each iteration, the label (score) for each newly acquired point is perturbed with a random value drawn from a normal distribution with standard deviation = 1.

#### Data enrichment
You can enrich the initial set with top-scoring molecules. In pharmacology, for instance, this might simulate already-known good binders or inhibitors. 

One way to do this is to manually pick top molecules from your dataset and specify them as seeds, combined with some random molecules. (Refer to: [Use SMILES as seeds for every experiment](#use-smiles-as-seeds-for-every-experiment).)

MDRMF also supports quick data enrichment via the following syntax. For example, here we select 3 molecules from the top 100–500 range in the dataset and combine them with 7 randomly chosen ones (for a total of 10 molecules):

```yaml
- unique_initial_sample:
    sample_size: 10
    nudging: [3, 100, 500]

- Experiment:
  # setup ...

- Experiment:
  # setup ...
```

#### Feature importance
You can enable feature importance optimization for **RF** by specifying:

```yaml
- Experiment:
    name: RF rdkit2D feature importance 20
    replicate: 30

    dataset: datasets/datasets.pkl

    model:
        name: RF
        iterations: 5
        acquisition_size: 20
        acquisition_method: greedy
        feature_importance_opt: {'iterations': 5, 'features_limit': 20}
```
This will train an RF model on all features and then, based on computed feature importances, run the active learning experiment with only the top features.

# Using experimental data (prospective study)
To run a prospective study (where you provide labels from real experiments), the setup is almost identical to the retrospective study, but you must use `labelExperiment` instead of `Experiment`. Example:

```yaml
- labelExperiment:
    name: prospective_docking_experiment

    data:
      datafile: unlabeled_data.csv
      SMILES_col: SMILES
      scores_col: measured_values
      ids_col: SMILES

    featurizer:
      name: morgan

    model:
        name: RF
        iterations: 5
        initial_sample_size: 10
        acquisition_size: 20
        acquisition_method: greedy

    metrics:
        names: [top-k-acquired]
        k: [100]
```

Your `.csv` file should contain a SMILES column and a score column. Before using the software the first time, you will have physically tested a random subset of molecules and entered their labels in the `.csv`. The software then builds a model from those labeled entries and suggests which molecules to test next.

An example `.csv` file might look like:
```
score,SMILES
1.12,C[C@@H](NC(=O)N1C[C@H](c2ccccc2)[C@H]2COCC[C@H]21)c1ccc(NC(=O)NC2CC2)cc1
,O=C(Nc1cccc(C(=O)N2CCC(c3c[nH]c4ncccc34)CC2)c1)[C@@H]1Cc2ccccc2O1
8.91,Cc1nn(-c2ccccc2)c2nc(C(=O)N3CCC([C@H]4C(=O)Nc5ccccc54)CC3)ccc12
3.15,Cc1cc(C)cc(C(=O)N2CCC[C@H](C(=O)NCc3cccc([C@@]4(C)NC(=O)NC4=O)c3)C2)c1
,CS(=O)(=O)c1ccc(F)c(C(=O)Nc2ccc(-c3nc(-c4ccccc4)n[nH]3)cc2)c1
,O=C1Nc2ccccc2[C@@H]1C1CCN(C(=O)c2cccc(N3C(=O)c4ccccc4C3=O)c2)CC1
5.11,NC(=O)[C@H]1CCCN(c2ccc(C(=O)N3CCC(c4cc5ccccc5[nH]4)CC3)cc2)C1
,Cn1c(=O)[nH]c2ccc(C(=O)NCC[C@H]3CN(c4ncnc5[nH]ncc45)c4ccccc43)cc21
,O=C(NCC(=O)N1CCc2ccccc2C1)[C@@H]1C[C@H](O)CN1C(=O)OCc1ccccc1
9.47,C#Cc1cc(F)c(NC(=O)C(=O)N2CC=C(c3c[nH]c4ncccc34)CC2)c(F)c1
[...more data]
```
Any entries with empty score fields (`score,SMILES`) mean they have not yet been experimentally tested, so the software will not use them for training until a label is added.