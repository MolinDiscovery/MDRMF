import yaml
import os
import pandas as pd
import inspect
import time
import shutil
from typing import List
from MDRMF.evaluator import Evaluator
import MDRMF.models as mfm
from MDRMF import Dataset, MoleculeLoader, Featurizer, Model

class Experimenter:

    def __init__(self, config_file: str):
        #self.config_file = config_file
        #self.experiments = self._load_config()

        self.config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), config_file)
        self.experiments = self._load_config()
        self.root_dir = "Experiment_Root"

        # Setting up root directory
        os.makedirs(self.root_dir, exist_ok=True)
    
        # Copy settings file to root dir
        destination_file_path = os.path.join(self.root_dir, "settings.yaml")
        shutil.copy(self.config_file, destination_file_path)

    def _load_config(self) -> List[dict]:
        with open(self.config_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return []

        # If there is only one experiment, make it into a list
        if isinstance(config, dict):
            config = [config]

        return [config]
    
    def conduct_all_experiments(self):

        start_time = time.time()
 
        for config in self.experiments:
            for experiment in config:
                key, value = list(experiment.items())[0]
                if key == 'Experiment':
                    self.conduct_experiment(value)
                if key == 'Dataset':
                    # Call self.make_dataset(value)
                    pass
                if key == 'Parallelize_experiments':
                    # add code here to handle 'Parallelize_experiments' cases
                    pass

        def _format_time(seconds):

            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            if hours:
                time_str = "{} hour(s), {} minute(s), {} second(s)".format(int(hours), int(minutes), int(seconds))
            elif minutes:
                time_str = "{} minute(s), {} second(s)".format(int(minutes), int(seconds))
            else:
                time_str = "{} second(s)".format(int(seconds))

            return time_str
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("Lab time over. All experiments conducted. Look for the results folder.")
        print("Time elapsed: ", _format_time(elapsed_time))
    
    def conduct_experiment(self, exp_config: dict):

        # --- Data setup --- #
        # If there is a dataset use this
        if 'dataset' in exp_config:
            dataset_file = exp_config['dataset']
            dataset_model = Dataset.load(dataset_file)
        elif 'data' in exp_config:
            # Load data
            data_conf = exp_config['data']

            datafile = data_conf['datafile']
            SMILES = data_conf['SMILES_col']
            scores = data_conf['scores_col']
            ids = data_conf['ids_col']

            data = MoleculeLoader(datafile, SMILES, scores).df

            # Featurize
            feat = Featurizer(data)
            feat_config = exp_config['featurizer']

            feat_type = feat_config['name']
            feat_params = feat_config.copy()
            del feat_params['name']

            features = feat.featurize(feat_type, **feat_params)

            # Get data
            X = features
            y = data[scores]
            ids_data = data[ids]

            # Make datasets
            dataset_model = Dataset(X=X, y=y, ids=ids_data)

            # Save the dataset
            dataset_model.save("dataset_" + exp_config['name']+".pkl")

        # --- Directory setup --- #
        # Create main directory
        experiment_directory = os.path.join(self.root_dir, exp_config['name'])
        os.makedirs(experiment_directory, exist_ok=True)

        # Save dataset
        dataset_file = os.path.join(experiment_directory, "dataset.pkl")
        dataset_model.save(dataset_file)

        # Create models directory
        models_directory = os.path.join(experiment_directory, "models")
        os.makedirs(models_directory, exist_ok=True)
        
        # --- Model setup --- #
        model_config = exp_config['model']
        model_name = model_config['name']
        model_params = model_config.copy()
        del model_params['name']

        # Check if model class exists
        model_class = None
        for name, obj in inspect.getmembers(mfm):
            if inspect.isclass(obj) and name == model_name:
                model_class = obj
                break

        if model_class is None:
            raise ValueError(f"Model {model_name} not found in MDRMF.models")

        # Setup evaluator
        model_metrics = exp_config['metrics']
        metrics = model_metrics['names']
        k_values = model_metrics['k']
        evaluator = Evaluator(dataset_model, metrics, k_values)

        results_list = []

        # --- Conduct replicate experiments and save results --- #
        for i in range(exp_config['replicate']):
            print(f"Running Experiment {exp_config['name']} replicate {i+1}")

            # Setup model
            model_input = model_class(dataset_model, evaluator=evaluator, **model_params)
            model = Model(model=model_input)
            model.train()
            
            # Save model
            model_file = os.path.join(models_directory, f"{model_name} Exp{i+1}.pkl")
            model.save(model_file)

            # Add results to list
            results = model.results
            for rank, score_dict in results.items():
                result_dict = {'replicate': i+1, 'rank': rank}
                result_dict.update(score_dict)
                results_list.append(result_dict)
            
        # Convert results to a DataFrame 
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(os.path.join(experiment_directory, "results.csv"), index=False)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Conduct machine fishing experiments based on a YAML config file.')
    parser.add_argument('config_file', type=str, help='The path to the YAML configuration file.')
    args = parser.parse_args()

    experimenter = Experimenter(args.config_file)
    experimenter.conduct_all_experiments()

    # To run an experiment after `pip install MDRMF` do this in your command prompt.
    # python -m MDRMF.experimenter config-file.yaml
    # An example config file is found in an example folder (not created yet)