from fileinput import filename
import pandas as pd
import numpy as np
import os
from deepchem import deepchem as dc

#####################
#      Classes      #
#####################
class DoNothingTransformer(dc.trans.Transformer):
    def __init__(self, **kwargs):
        super(DoNothingTransformer, self).__init__(**kwargs)

    def transform_array(self, X, y, w, ids):
        """
        This method does nothing and returns the input data as it is. Use this in case you don't want to have your data transformed.

        Parameters
        ----------
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Labels
        w : numpy.ndarray
            Weights
        ids : numpy.ndarray
            IDs

        Returns
        -------
        tuple
            A tuple containing the unchanged input data (X, y, w, ids)
        """
        
        return X, y, w, ids
    
    def untransform(self, data):
        """
        This method does nothing and returns the input data as it is.

        Parameters
        ----------
        data : numpy.ndarray
            Data that was previously transformed (or not transformed) by this class

        Returns
        -------
        numpy.ndarray
            The unchanged input data
        """
        return data


#####################
#     Functions     #
#####################

def load_data(datafile: str, MolWt: int, first_index: bool) -> pd.DataFrame:
    """
    Loads molecular data from a CSV file, processes it, and returns a pandas DataFrame with a specified molecular weight threshold.
    The dataset must contain a column called SMILES with smile strings.

    Parameters:
    - datafile (str): The path to the input CSV data file containing molecular data.
    - MolWt (int): The molecular weight threshold. Molecules with molecular weight less than this threshold will be included in the resulting DataFrame.
    - first_index (bool): A boolean flag indicating if the first column of the input CSV file should be used as the index of the resulting DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing the processed molecular data.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    df = pd.read_csv(datafile)

    if first_index == True:
        df = df.set_index(df.columns[0])

    if 'molecules' not in df.columns:
        df['molecules'] = df['SMILES'].apply(Chem.MolFromSmiles)

    if 'MolWt' not in df.columns:
        df['MolWt'] = df['molecules'].apply(Descriptors.MolWt)

    # Determines what size of molecules should be included.
    df = df[df['MolWt'] < MolWt]

    return df


def receptor_data(df,receptor_name, featurizer=dc.feat.ConvMolFeaturizer(), seed=1, input_transformer=dc.trans.NormalizationTransformer):
    '''
    This function takes a DataFrame containing data for multiple receptors, processes it, and returns training, validation, and test datasets that can be used in a DeepChem model.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing data for all receptors.
    - receptor_name (str): The name of the receptor for which the data is to be processed.
    - featurizer (dc.feat): The featurizer to use. Defaults to ConvMolFeaturizer(). To use another the use MolGraphConvFeaturizer use dc.feat.MolGraphConvFeaturizer().
    - seed (int, optional): The random seed for splitting the data into train, validation, and test sets. Default is 1.
    - transformer (dc.trans.NormalizationTransformer, optional): set the DeepChem transformer to use. Defaults to NormalizationTransformer.
      Please note you can set this to `transformer=utils.DoNothingTransformer` which avoids any transformation altogether;
      assuming you imported this file as utils.
      Also note that the transformer transforms the y-values not the X-values.
    
    Returns:
    - train_dataset (dc.data.NumpyDataset): A DeepChem NumpyDataset object containing the training data.
    - valid_dataset (dc.data.NumpyDataset): A DeepChem NumpyDataset object containing the validation data.
    - test_dataset (dc.data.NumpyDataset): A DeepChem NumpyDataset object containing the test data.
    - transformer (dc.trans.NormalizationTransformer): A DeepChem NormalizationTransformer object that was used to transform the output data for regression.

    Notes: Importantly also removes any 0 values. (which are usually artificially assigned to non-docked ligands)
    '''
    
    # This code takes the initial df and republish it with data only on one receptor
    df_receptor =  df[[receptor_name, 'molecules', 'SMILES']]
    # Applying a mask to remove possible 0 values.
    mask = df_receptor[receptor_name] == 0.0
    df_receptor = df_receptor.drop(df[mask].index)

    feat = featurizer
    X = feat.featurize(df_receptor['molecules'])
    y = df_receptor[receptor_name].to_numpy()
    ids = df_receptor['SMILES']

    # Create dataset for deepchem
    dataset = dc.data.NumpyDataset(X=X, y=y, ids=ids)

    # Transform the output data for regression
    transformer = input_transformer(transform_y=True, dataset=dataset)
    dataset = transformer.transform(dataset)

    # split data
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset = dataset,
    frac_train = 0.8,
    frac_valid = 0.1,
    frac_test = 0.1,
    seed=seed,
    )

    return train_dataset, valid_dataset, test_dataset, transformer


def set_seed(seed, tensorflow=True, pytorch=True):
    """
    Sets the random seed for various libraries (NumPy, TensorFlow, PyTorch, and Python's random module).

    Parameters:
    - seed (int): The random seed value to set for the libraries.
    - tensorflow (bool, optional): Set the seed for TensorFlow if True. Defaults to True.
    - pytorch (bool, optional): Set the seed for PyTorch if True. Defaults to True.

    """
    # Set seed for TensorFlow
    try:
        if tensorflow:
            import tensorflow as tf
            tf.random.set_seed(seed)
    except:
        print("Please import Tensorflow as tf to set its seed.")
    
    # Set seed for PyTorch
    try:    
        if pytorch:
            import torch
            torch.manual_seed(seed)

            # Set seed for PyTorch's CUDA and enforce deterministic behavior
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except:
        print("Please import PyTorch to set its seed.")
    
    # Set seed for NumPy and Python's random module
    try:
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
    except:
        print("You must import numpy as np and random to set up seeds.")


def fit_best_pt(model, train_dataset, valid_dataset, metric, transformers, nb_epoch=100, patience=3, interval=1, high_is_better=True, model_name="model"):
    """
    Train a deepchem(PyTorch) model using early stopping based on the performance on a validation dataset.
    Please use the fit_best_tf for tensorflow models.
    
    Parameters:
    - model: The model to be trained.
    - train_dataset (dc.data.Dataset): The training dataset.
    - valid_dataset (dc.data.Dataset): The validation dataset.
    - metric (dc.metrics.Metric): The metric used to evaluate the model's performance.
    - transformers (list): A list of transformers applied to the dataset.
    - nb_epoch (int, optional): The maximum number of epochs for training. Defaults to 100.
    - patience (int, optional): The number of epochs to wait without improvement before stopping the training. Defaults to 3.
    - interval (int, optional): The interval (in epochs) between validation checks. Defaults to 1.
    - high_is_better(bool, optional): Set this to True if higher scores are better (R2) or False if low scores are better (RMSE). Default to True.
    - model_name (str, optional): The name used to when saving the model. Defaults to "model".
    
    Returns:
    - list: A list of tuples containing the epoch number, validation score, and training score for each validation epoch.

    Note: Use only for PyTorch deepchem models (Not RF, tensorflow etc.). Please use the fit_best_tf for tensorflow models.
    """
    import copy

    def get_unique_model_filename(prefix=model_name, suffix=".ckpt"):
        counter = 1
        while True:
            filename = f"{prefix}{counter:02d}{suffix}"
            if not os.path.exists(os.path.join("models", filename)):
                return filename
            counter += 1

    best_model = None
    best_score = None
    best_epoch = None
    list_scores = []
    wait = 0

    for epoch in range(nb_epoch):
        print(f"Epoch {epoch+1}/{nb_epoch}")
        model.fit(train_dataset, nb_epoch=1)

        if (epoch + 1) % interval == 0:
            valid_scores = model.evaluate(valid_dataset, metric, transformers)
            valid_score = valid_scores[metric[0].name]
            print(valid_scores)

            training_scores = model.evaluate(train_dataset, metric, transformers)
            training_score = training_scores[metric[0].name]

            list_scores.append((epoch + 1, valid_score, training_score))

            if high_is_better:
                condition = best_score is None or valid_score > best_score
            else:
                condition = best_score is None or valid_score < best_score            

            if condition:
                best_score = valid_score
                best_epoch = epoch + 1
                best_model = copy.deepcopy(model)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered at epoch:", epoch + 1)
                    break
    
    print(f"Best model found at epoch {best_epoch} with {metric[0].name} score: {best_score}")

    unique_filename = get_unique_model_filename()
    os.makedirs("models", exist_ok=True)
    best_model.save_checkpoint(model_dir=os.path.join("models", unique_filename))

    return list_scores


def fit_best_tf(model, train_dataset, valid_dataset, metric, transformers, nb_epoch=100, patience=3, interval=1, high_is_better=True, model_name="model"):
    """
    Train a TensorFlow model using early stopping based on the performance on a validation dataset.

    Parameters:
    - model: The TensorFlow model to be trained.
    - train_dataset (dc.data.Dataset): The training dataset.
    - valid_dataset (dc.data.Dataset): The validation dataset.
    - metric (dc.metrics.Metric): The metric used to evaluate the model's performance.
    - transformers (list): A list of transformers applied to the dataset.
    - nb_epoch (int, optional): The maximum number of epochs for training. Defaults to 100.
    - patience (int, optional): The number of epochs to wait without improvement before stopping the training. Defaults to 3.
    - interval (int, optional): The interval (in epochs) between validation checks. Defaults to 1.
    - high_is_better(bool, optional): Set this to True if higher scores are better (R2) or False if low scores are better (RMSE). Default to True.
    - model_name (str, optional): The name used when saving the model. Defaults to "model".
    
    Returns:
    - list: A list of tuples containing the epoch number, validation score, and training score for each validation epoch.

    Note: Use only for deepchem models using TensorFlow (Not suitable for PyTorch, RandomForest, etc.). Please use the fit_best_pt for PyTorch models.
    """
    
    def get_unique_model_filename(prefix=model_name, suffix=".ckpt"):
        counter = 1
        while True:
            filename = f"{prefix}{counter:02d}{suffix}"
            if not os.path.exists(os.path.join("models", filename)):
                return filename
            counter += 1

    def get_best_model_weights(model):
        return model.model.get_weights()

    def set_best_model_weights(model, best_model_weights):
        model.model.set_weights(best_model_weights)

    best_score = None
    best_epoch = None
    best_model_weights = None
    list_scores = []
    wait = 0

    for epoch in range(nb_epoch):
        print(f"Epoch {epoch+1}/{nb_epoch}")
        model.fit(train_dataset, nb_epoch=1)

        if (epoch + 1) % interval == 0:
            valid_scores = model.evaluate(valid_dataset, metric, transformers)
            valid_score = valid_scores[metric[0].name]
            print(valid_scores)

            training_scores = model.evaluate(train_dataset, metric, transformers)
            training_score = training_scores[metric[0].name]

            list_scores.append((epoch + 1, valid_score, training_score))

            if high_is_better:
                condition = best_score is None or valid_score > best_score
            else:
                condition = best_score is None or valid_score < best_score

            if condition:
                best_score = valid_score
                best_epoch = epoch + 1
                best_model_weights = get_best_model_weights(model)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered at epoch:", epoch + 1)
                    break

    print(f"Best model found at epoch {best_epoch} with {metric[0].name} score: {best_score}")

    set_best_model_weights(model, best_model_weights)

    unique_filename = get_unique_model_filename()
    os.makedirs("models", exist_ok=True)
    model.save_checkpoint(model_dir=os.path.join("models", unique_filename))

    return list_scores


def plot_predictions(model, training_data, test_data, transformer):
    from matplotlib import pyplot as plt

    train_preds = model.predict(training_data, [transformer]).flatten()
    test_preds = model.predict(test_data, [transformer]).flatten()
    train_plot = transformer.untransform(training_data.y)
    test_plot = transformer.untransform(test_data.y)

    plt.plot(train_plot, train_plot, label='True values')
    plt.scatter(train_plot, train_preds, marker='.', label='train preds')
    plt.scatter(test_plot, test_preds, marker='.', label='test preds')
    plt.gca().set(xlabel='Docking scores (true)', ylabel='Docking scores(predict)', title='Predictions')
    plt.legend()


def plot_validation(trained_model, metric):
    from matplotlib import pyplot as plt
    
    epochs = [x[0] for x in trained_model]
    valid_scores = [x[1] for x in trained_model]
    train_scores = [x[2] for x in trained_model]

    plt.plot(epochs, train_scores, label='Training')
    plt.plot(epochs, valid_scores, label='Validation')
    plt.gca().set(xlabel='Epochs', ylabel=metric.name, title='Validation')
    plt.xticks(range(min(epochs), max(epochs) + 1, 4))
    plt.legend()
    plt.show

def eval(model, test_data, transformer=[]):
    metrics = [
        dc.metrics.Metric(dc.metrics.r2_score),
        dc.metrics.Metric(dc.metrics.rms_score),
        dc.metrics.Metric(dc.metrics.mean_absolute_error),
    ]

    dc_eval = model.evaluate(test_data, metrics, transformer)
    
    rmse = dc_eval.get('rms_score')
    r2 = dc_eval.get('r2_score')
    mae = dc_eval.get('mean_absolute_error')
    
    preds = model.predict(test_data, transformer).flatten()
    mean = np.mean(preds)
    std = np.std(preds)

    score_names = ['RMSE', 'R2  ', 'MAE ', 'mean', 'std ']
    scores_values = [rmse, r2, mae, mean, std]
    scores = zip(score_names, scores_values)

    for i, j in scores:
        print(i, "  |", round(j, 3))

def save_rf_model(model, model_name):
    from joblib import dump

    def get_unique_model_filename(prefix=model_name, suffix=".joblib"):
        counter = 1
        while True:
            filename = f"{prefix}{counter:02d}{suffix}"
            if not os.path.exists(os.path.join("models", filename)):
                return filename
            counter += 1

    unique_filename = get_unique_model_filename()
    os.makedirs("models", exist_ok=True)
    dump(model.model, os.path.join("models", unique_filename))