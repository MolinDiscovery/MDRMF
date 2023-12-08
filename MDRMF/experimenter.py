import numpy as np
import yaml
import os
import pandas as pd
import inspect
import time
import shutil
import datetime
import uuid
import json
from typing import List
from MDRMF.evaluator import Evaluator
import MDRMF.models as mfm
from MDRMF import Dataset, MoleculeLoader, Featurizer, Model

class Experimenter:

    def __init__(self, config_file: str):

        self.config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), config_file)
        self.experiments = self._load_config()
        self.uniform_initial_sample = self.get_config_value(['uniform_initial_sample']) or None
        self.unique_initial_sample = self.get_config_value(['unique_initial_sample', 'sample_size']) or None
        self.save_models = self.get_config_value(['save_models']) or False # Don't save models by default.
        self.save_datasets = self.get_config_value(['save_datasets']) or False # Don't save datasets by default.
        self.save_nothing = self.get_config_value(['save_nothing']) or False # Save results by default, if True deletes all data after completion.

        # Generate ids
        self.protocol_name = self.get_protocol_name()
        id = self.generate_id(self.protocol_name)

        # Setting up root directory
        self.root_dir = id
        os.makedirs(self.root_dir, exist_ok=True)
    
        self.create_meta_data()


    def get_protocol_name(self) -> str:
        try:
            return self.experiments[0][0]['Protocol_name']
        except KeyError as exc:
            return "protocol"


    def generate_id(self, protocol_name: str) -> str:
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%y%m%d-%H%M%S')  # YYMMDD-HHMMSS
        # Create a UUID and take the first 8 characters for a shorter hash
        uuid_hash = str(uuid.uuid4())[:8]
        id = f"{protocol_name}-{formatted_time}-{uuid_hash}"
        return id


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


    def create_meta_data(self):
        destination_file_path = os.path.join(self.root_dir, "settings.yaml")
        shutil.copy(self.config_file, destination_file_path)

        meta_destination = os.path.join(self.root_dir, "meta_data.txt")
        with open(meta_destination, "w") as f:
            f.write(self.root_dir)


    def cleanup(self):
        """
        Delete the entire root directory if save_nothing is True.
        """
        if hasattr(self, "save_nothing") and self.save_nothing is True:
            try:
                shutil.rmtree(self.root_dir)
                print(f"Lab time over. Successfully deleted the directory: {self.root_dir}")
                print("To change this behavior set ´- save_nothing: False´ or delete the line entirely.")
            except Exception as e:
                print(f"Error deleting the directory: {self.root_dir}. Error: {e}")


    def get_config_value(self, keys: List[str]):
        """
        This method iterates over the list of experiments to find the 
        specified key and its sub-keys, and returns the associated value.
        
        The method navigates through any number of sub-levels to access 
        the value of the deepest sub-key.
        
        If the specified key or any of the sub-keys are not found within 
        the experiments, the method returns None.
        
        Parameters:
            keys (List[str]): A list where the first item is the top-level 
                            key and each subsequent item is a sub-key 
                            at the next level down.
        
        Returns:
            The value associated with the deepest sub-key if present, 
            otherwise None.
        
        Example:
            If the YAML file contains:
                - top-key:
                    sub-key: 10
                
            Then calling get_config_value(['top-key', 'sub-key']) will return 10.
        """
        for config in self.experiments:
            for entry in config:
                value = entry  # starting point is the dictionary itself
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        break  # if key is not in the dict, break out of the loop
                else:
                    # if the for loop didn't break, the deepest value was found
                    return value
        return None


    def _get_or_create_dataset(self, exp_config: dict, semi_labeled=False):
        if 'dataset' in exp_config:
            dataset_file = exp_config['dataset']
            return Dataset.load(dataset_file)
        elif 'data' in exp_config:
            data_conf = exp_config['data']
            datafile = data_conf['datafile']
            SMILES = data_conf.get('SMILES_col', None)
            vector = data_conf.get('vector_col', None)
            scores = data_conf['scores_col']
            ids = data_conf['ids_col']
            
            if vector:
                data = pd.read_csv(datafile)
                data[vector] = data[vector].apply(json.loads) # deserialize json strings to lists
                X = data[vector]
                y = data[scores]
                ids_data = data[ids]
            else:
                data = MoleculeLoader(datafile, SMILES, scores).df
            
                # Featurize
                feat = Featurizer(data)
                feat_config = exp_config.get('featurizer', {})
                feat_type = feat_config.get('name', '')
                feat_params = feat_config.copy()
                del feat_params['name']
                
                features = feat.featurize(feat_type, **feat_params)
                
                # Get data
                X = features
                y = data[scores]
                ids_data = data[ids]
            
            # Make datasets
            if semi_labeled == True:
                # When the arg ´semi_labeled´ is False, all NaN values is removed
                # thus keeping only the labeled data.                
                dataset_labeled = Dataset(X=X, y=y, ids=ids_data, keep_unlabeled_data_only=False)
                # When it is True only 
                dataset_unlabeled = Dataset(X=X, y=y, ids=ids_data, keep_unlabeled_data_only=True)
                return dataset_labeled, dataset_unlabeled
            else:
                dataset_model = Dataset(X=X, y=y, ids=ids_data)
                return dataset_model
        else:
            return None


    def _calculate_uniform_indices(self, exp_config: dict, uniform_sample: int):
        if 'dataset' in exp_config:
            dataset_file = exp_config['dataset']
            dataset = Dataset.load(dataset_file)
            length = dataset.get_length()
            random_indices = np.random.randint(0, length, uniform_sample).tolist()
            return random_indices
        elif 'data' in exp_config:
            data_conf = exp_config['data']
            datafile = data_conf['datafile']
            SMILES = data_conf['SMILES_col']
            scores = data_conf['scores_col']
            ids = data_conf['ids_col']

            data = MoleculeLoader(datafile, SMILES, scores).df
            length = len(data)
            random_indices = np.random.randint(0, length, uniform_sample).tolist()
            return random_indices
        
    
    def _calculate_unique_indices(self, exp_config: dict, unique_sample_size: int, nudging: list = []):

        '''
        Maybe I need to make the method return the ids and not the mere indices.
        Doing the nudging like now where we just sort and return a indices
        for the sorted data will also require the working dataset to be sorted,
        which might not always be the case. Currently most datasets are sorted
        by y though, but it can be a future problem.
        '''
        
        # divide the total number of unique samples in two to get
        nudged_samples_size = int(unique_sample_size / 2)

        nudge_size = nudging[0]
        nudge_top_n = nudging[1]

        if 'dataset' in exp_config:
            dataset_file = exp_config['dataset']
            dataset = Dataset.load(dataset_file)
            if nudging != []:
                dataset.sort_by_y()
                top_dataset = dataset.get_top_or_bottom(nudge_top_n)
                _, random_top_indices = top_dataset.get_samples(nudge_size, return_indices=True)
                n_non_top_indices = unique_sample_size-nudge_size
                _, random_indices = dataset.get_samples(n_non_top_indices, return_indices=True)
                indices = np.concatenate([random_top_indices, random_indices])
            else:
                _, indices = dataset.get_samples(unique_sample_size)

        elif 'data' in exp_config:
            data_conf = exp_config['data']
            datafile = data_conf['datafile']
            SMILES = data_conf['SMILES_col']
            scores = data_conf['scores_col']
            ids = data_conf['ids_col']
            data = MoleculeLoader(datafile, SMILES, scores).df
            length = len(data)
            if nudging != []:
                data.sort_values(by=[scores])
                top_data = data.iloc[:nudge_top_n]
                random_top_indices = np.random.randint(0, len(top_data), nudged_samples_size).tolist()
                random_indices = np.random.randint(0, length, nudged_samples_size).tolist()

                indices = random_top_indices + random_indices
            else:
                indices = np.random.randint(0, length, unique_sample_size).tolist()

        return indices


    def conduct_all_experiments(self):
        start_time = time.time()

        uniform_indices = None

        # if user inputs a uniform_initial_sample use this to 'seed' the model.
        if self.uniform_initial_sample is not None:
            # Assuming the first experiment configuration is representative for the initial uniform sample
            first_experiment_config = next((exp.get('Experiment', {}) for exp in self.experiments[0] if 'Experiment' in exp), {})
            
            uniform_indices = self._calculate_uniform_indices(first_experiment_config, self.uniform_initial_sample)
            print("Uniform Indices:", uniform_indices)


        '''
        Below code creates unique initial_samples. It uses the first experiment to retrieve how many replicates to create.
        '''

        if self.unique_initial_sample is not None:

            first_experiment_config = next((exp.get('Experiment', {}) for exp in self.experiments[0] if 'Experiment' in exp), {})

            nudge_value = self.get_config_value(['unique_initial_sample', 'nudging'])

            if isinstance(nudge_value, list):
                nudging = nudge_value
            elif nudge_value == None:
                pass
            else:
                ValueError('Problem reading nudge values. It must be a list consisting of how many samples to include\n'
                           'and what top_n space to draw from. Like [5, 100]\n'
                           'draws 5 random molecules from the top-100 best molecules.')

            replicates_first_exp = first_experiment_config['replicate']

            unique_indices_list = []

            for i in range(replicates_first_exp):

                unique_indices = self._calculate_unique_indices(first_experiment_config, self.unique_initial_sample, nudging)
                unique_indices_list.append(unique_indices)

        '''
        End creating unique initial sample.
        '''

        for config in self.experiments:
            for experiment in config:
                key, value = list(experiment.items())[0]
                if key == 'Experiment':
                    self.conduct_experiment(value, uniform_indices, unique_indices_list)
                elif key == 'Dataset':
                    # Call self.make_dataset(value)
                    pass
                elif key == 'labelExperiment':
                    self.conduct_labelExperiment(value)

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

        if hasattr(self, "save_nothing") and self.save_nothing is True:
            self.cleanup()
        else:
            print("Lab time over. All experiments conducted. Look for the results folder.")

        print("Time elapsed: ", _format_time(elapsed_time))
    

    def conduct_experiment(self, exp_config: dict, uniform_indices=None, unique_indices=None):

        dataset_model = self._get_or_create_dataset(exp_config)
        if not dataset_model:
            raise ValueError("Unable to create or load a dataset model.")

        # --- Directory setup --- #
        experiment_directory = os.path.join(self.root_dir, exp_config['name'])
        os.makedirs(experiment_directory, exist_ok=True)
        
        # Save dataset
        if self.save_datasets is True:
            dataset_file = os.path.join(experiment_directory, "dataset.pkl")
            dataset_model.save(dataset_file)

        # Create models directory
        if self.save_models is True:
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

        # If unique indices are defiend use the length of the list to define how many replicates to do.
        if unique_indices is not None:
            replicates = len(unique_indices)
        else:
            replicates = exp_config['replicate'] # If replicates is not predefined get replicate from current experiment.

        results_list = []

        # --- Conduct replicate experiments and save results --- #
        for i in range(replicates):
            print(f"Running Experiment {exp_config['name']} replicate {i+1}")

            # get a fresh copy of dataset_model for each replicate
            dataset_model_replicate = dataset_model.copy()

            # Setup model
            if uniform_indices is not None:
                model_input = model_class(dataset_model_replicate, evaluator=evaluator, seeds=uniform_indices, **model_params)
            elif unique_indices is not None:
                unique_seeds = unique_indices[i]
                model_input = model_class(dataset_model_replicate, evaluator=evaluator, seeds=unique_seeds, **model_params)
            else:
                model_input = model_class(dataset_model_replicate, evaluator=evaluator, **model_params)
            model = Model(model=model_input)
            model.train()
            
            # Save model
            if self.save_models is True:
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


    def conduct_labelExperiment(self, exp_config: dict):
        dataset_labeled, dataset_unlabeled = self._get_or_create_dataset(exp_config, semi_labeled=True)

        model_config = exp_config['model']
        model_name = model_config['name']
        model_params = model_config.copy()
        del model_params['name']

        model_class = None
        for name, obj in inspect.getmembers(mfm):
            if inspect.isclass(obj) and name == model_name:
                model_class = obj
                break

        if model_class is None:
            raise ValueError(f'Model {model_name} not found in MDRMF.models')
        
        length_labeled_data = dataset_labeled.y.shape[0]

        model_input = model_class(
            dataset_labeled,
            iterations=0,
            initial_sample_size=length_labeled_data,
            **model_params
            )
        
        model = Model(model=model_input)
        model.train()
        acquired_points = model.get_acquired_points(dataset_unlabeled)
        print(acquired_points)
        

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