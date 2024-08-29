# Table of Contents

- [Table of Contents](#table-of-contents)
- [MDRMF](#mdrmf)
- [Why is it called MDRMF?](#why-is-it-called-mdrmf)
- [What does MDRMF do?](#what-does-mdrmf-do)
- [Installation](#installation)
- [How to use MDRMF](#how-to-use-mdrmf)
  - [Testing your setup (retrospective study)](#testing-your-setup-retrospective-study)
    - [data](#data)
    - [featurizer](#featurizer)
    - [model](#model)
    - [metrics](#metrics)
    - [Configuration documentation](#configuration-documentation)
      - [Create multiple experiments](#create-multiple-experiments)
      - [Create a pre-featurized dataset](#create-a-pre-featurized-dataset)
      - [Use a prefeaturized dataset](#use-a-prefeaturized-dataset)
      - [Using a prefeaturized CSV file.](#using-a-prefeaturized-csv-file)
      - [Use SMILES as seeds to for every experiment experiments.](#use-smiles-as-seeds-to-for-every-experiment-experiments)
      - [Generate random seeds, but use the same random seeds for every experiment.](#generate-random-seeds-but-use-the-same-random-seeds-for-every-experiment)
      - [Acquisition functions](#acquisition-functions)
      - [Adding noise to the data](#adding-noise-to-the-data)
      - [Data enrichment](#data-enrichment)
      - [Feature importance](#feature-importance)


# MDRMF
Multiple drug resistance machine fishing
- A project to explore ML models using active pool-based learning (machine fishing) for transporters implicated in MDR.

# Why is it called MDRMF?
MDRMF is a python package that was developed as part a project to find inhibitors of ABC transporters which cause multiple-drug resistance towards chemotherapeutics. The machine fishing part refers to the fact that active learning can be seen as fishing for drug candidates in a sea of molecules.

# What does MDRMF do?
MDRMF is a platform that can help find candidate drugs for a particular disease target. The software have two modes. 
1) A retrospective mode for testing and optimization purposes of the active learning workflow.
2) A prospective mode for usage in experimental settings.

**Retrospective part:** This is for testing and optimization purposes. You have dataset with SMILESs that is fully labelled with some score (e.g. docking). The software can evaluate how many hits its able to obtain with the specified settings.

**Prospective idea:** The software was designed to be used on experimental data. That is, you have a list of SMILESs from the dataset you're investigating. You select X number of molecules from the dataset and test them to obtain labels. These labels are assigned to the corresponding molecules and in given to the software for training. The software will then return X number of molecules that it wants you to test next.

# Installation
```bash
pip install MDRMF
````

Ensure the required dependancies are installed. Preferentially create a conda environment from environment.yaml.
```bash
conda env create -f environment.yaml
```
This will also install MDRMF it self.

# How to use MDRMF
MDRMF works by reading YAML configuration files that defines experiments you want to do. When you run an experiment it will create a directory with the name of the configuration file. In here results for each experiment will be saved along with other miscellaneous information such as the training datasets, graphs, settings and more.

## Testing your setup (retrospective study)
Here is an example of the most simple setup.

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

Here, one experiment will be conducted named `retrospective_docking_experiment`.

### data
This experiment reads a .csv file where two required things are specified - a `SMILES_col` and a `score_col`. Finally, an optional `ids_col` is set to the SMILES column in the .csv file (a sequantial list of numbers will be generated for ids if left unspecified).

### featurizer
The featurizer specifies how to describe the molecules. The following featurizers are currently supported.

```python
morgan, topological, MACCS, avalon, rdk, pharmacophore, rdkit2D, mqn
```
All of these are implementations from RDKit and you can directly pass argument to them. If you want to featurize with Morgan, but using a different bit vector length you can pass a function argument like this.
```yaml
    featurizer:
      name: morgan
      nBits: 2048
```

### model
This part defines the machine learning model to use and any active learning specifications you have.
Above we specify a random forest model to be initialized with 10 random molecules from the dataset while acquiring 20 new molecules for 5 iterations. The models are all from the sklearn package except for the LightGBM model. As with the featurizer you can pass arguments to the underlying model. Lets say our computer is equipped with a juicy multicore CPU we can pass the `n_jobs` argument to the underlying model.

```yaml
    model:
        name: RF
        iterations: 5
        initial_sample_size: 10
        acquisition_size: 20
        acquisition_method: greedy,
        n_jobs: 14 # (define no. of cores)
````
Currently, the following models are supported. 
```python
RF (random forest), MLP (Multi-layer perceptron), KNN (K-nearest neighbour), LGBM (LightGBM), DT (DecisionTree), SVR (Support Vector Regressor)
```
⚠️ Please note that MDRMF only use regression models, and classification is not supported.

### metrics
Metrics define how to evaluate the active learning experiment. In this demo setup we evaluate, at each iteration, how many of the highest 100 scored molecules was included in the training dataset. The full list of evaluators implemented:
```yaml
[top-k-acquired, R2_model, R2_k, top-k]
```
You can do multiple evaluations at each run by supplying a longer list `[100, 1000]`.

Notes: `top-k` returns how many of the top-k molecules the trained model predicts to be actual top k molecules. For instance, if you have a dataset of _mol1_ .. _mol100_ where the lower numbered molecules are higher scored and we set the metrics to be top-k with k=10. Then, if the models predicts _mol2_, _mol4_, _mol1_, _mol9_ as  5, 3, 8, 2, but predicts _mol3_, _mol5_, _mol6_, _mol7_, _mol8_ and _mol10_ as 19, 25, 11, 32, 25, 15, then this would evaluate to a top-k of 4/10 (0.4). This might give a measure of how well the model is able to predict in the region of interest.

ℹ️ Using R2_model, R2_k and top-k requires an additional prediction during AL-iterations. Prediction is resource intensive when using the pairwise algorithm and so it is advised not to use these when conducting pairwise (PADRE) experiments.

### Configuration documentation
#### Create multiple experiments
```yaml
- Experiment:
    name: exp1

setup ...

- Experiment:
    name: exp2

setup ...

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
This will create a pickle file that can be used for conducting experiments.
#### Use a prefeaturized dataset
```yaml
- Experiment:
    name: exp-using-dataset

    dataset: path/to/dataset.pkl

    model:
        ...

    metrics: 
        ...
```
Notice, that you don't need featurize, because the pickle dataset already contains the featurized molecules.

#### Using a prefeaturized CSV file.
MDRMF also accepts featurized molecules from a CSV file. The smart thing about this is that you can do your own featurizations and put them directly into MDRMF.
```yaml
- Experiment:
    name: retrospective_docking_experiment

    data:
      datafile: docking_data.csv
      vector_col: features
      scores_col: docking_score
      ids_col: SMILES

    ...
```
You can also just create a dataset directly using the `create_dataset` keyword instead of `Experiment`. <br>
See: [Create a pre-featurized dataset](#create-a-pre-featurized-dataset)

#### Use SMILES as seeds to for every experiment experiments.
These seeds will be used as the initial sample for all experiments. Using this setting will override the `initial_sample_size` argument in the modeller setup. The below setup would do two replicates of every experiment with the specificed seeds.
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
    ],
    ...,
  ]

- Experiment:
  setup ...

- Expderiment:
  setup ...

```
#### Generate random seeds, but use the same random seeds for every experiment.
Using this setting will override the `initial_sample_size` argument in the modeller setup.
```yaml
- unique_initial_sample:
    sample_size: 10

- Experiment:
  setup ...

- Expderiment:
  setup ...

```

#### Acquisition functions
Active learning is, like many low data machine learning scenarios, a balance between between exploring or exploiting your knowledge about the data.

Seven acquisition functions have been implemented in MDRMF.
```python
'greedy', 'MU' (most uncertainty), 'LCB' (Lower confidence bound), 'EI' (Expected improvement), 'TS' (Thompson sampling), 'tanimoto', and 'random'.
```
You simply pick the one to use in the model setup
```yaml
- Experiment:
    ...
    model:
        name: RF
        iterations: 5
        initial_sample_size: 10
        acquisition_size: 20
        acquisition_method: greedy # or MU, LCB, EI, TS, tanimoto, random
    ...
```
Please note that only RF, KNN and LGBM can use MU, LCB, EI or TS because these are the only models where an uncertainty is calculated along with each prediction.

####  Adding noise to the data
To simulate a prospective study (for example an _in vitro_ study) you can add noise to the data like this
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
This will add noise to the score (y-value) or each selected point to be trained on where each value is perturbed with a random number drawn from a normal distribution with a standard deviation of 1.

#### Data enrichment
You can enrich the initial set with top scoring molecules. In the context of pharmacology this can be used to simulate a scenario where you have some knowledge of good inhibitors or good binders your protein of interest.

One way to do this, and be in complete control is to inspect your dataset and find top-scored molecules. Then seed X number of these combined with som random selections. Refer to: [Use SMILES as seeds to replicate experiments.](#use-smiles-as-seeds-to-replicate-experiments)

MDRMF also supports a quick way to test data enrichment.
Here is an example where we select 3 molecules from best 100-500 molecules with a total sample size of 10. The 7 remaining molecules are drawn randomly from the entire dataset.
```yaml
- unique_initial_sample:
    sample_size: 10
    nudging: [3, 100, 500]

- Experiment:
  setup ...

- Expderiment:
  setup ...
```

#### Feature importance
You can declare a feature importance optimization where upon you can declare number of iterations and number of features. This has only implemented for RF, however. This will train the RF model, first on all features, and then afterwards the the AL-experiment will be conducted using a model that only considers the most important features.
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