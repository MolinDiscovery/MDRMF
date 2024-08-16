import os
import sys
import yaml
from pykwalify.core import Core

if sys.version_info >= (3, 7):
    from importlib.resources import read_text
else:
    from pkg_resources import resource_string

class ConfigValidator:
    
    def __init__(self) -> None:
        pass

    def load_yaml(self, file_path):
        """
        Loads YAML file and returns its content.
        Converts single experiment configs to a list format.
        """        
        with open(file_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return []
            
        # If there is only one experiment, make it into a list
        if isinstance(config, dict):
            config = [config]

        return config
    

    def load_schema(self, schema_name):
        """Load and return the YAML schema from within the package."""
        if sys.version_info >= (3, 7):
            schema = read_text('MDRMF.schemas', schema_name)
        else:
            schema = resource_string('MDRMF.schemas', schema_name).decode('utf-8')
        return yaml.safe_load(schema)


    def check_for_exps(self, config):
        """
        Validates the config for exclusive presence of 'Experiment', 'labelExperiment', or 'create_dataset'.
        Raises exceptions for invalid or conflicting configurations.
        """
        e_set = set()
        for i in config:
            if not isinstance(i, dict):
                raise ValueError(f'This top-level key is not accepted: {i}')
            
            for j in i.keys():
                e_set.add(j)

        if ('Experiment' in e_set and 'labelExperiment' in e_set) or ('Experiment' in e_set and 'create_dataset' in e_set) or ('labelExperiment' in e_set and 'create_dataset' in e_set):
            raise Exception('You cannot conduct "Experiment", "labelExperiment", or "create_dataset" simultaneously!')
        elif 'Experiment' in e_set or 'labelExperiment' in e_set or 'create_dataset' in e_set:
            pass
        else:
            raise Exception('''
    Fatal error while reading the config file.
    Please, include only one "Experiment", "labelExperiment", or "create_dataset",
    and check the structure of your config file.
                            ''')


    def check_dataset_filepath(self, file_path):
        return os.path.exists(file_path) and file_path.endswith(('.pkl', '.pickle'))
    

    def check_data_filepath(self, file_path):
        return os.path.exists(file_path) and file_path.endswith(('.csv'))


    def query_yes_no(self, question, default="yes"):
        """Ask a yes/no question via raw_input() and return their answer.

        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
                It must be "yes" (the default), "no" or None (meaning
                an answer is required of the user).

        The "answer" return value is True for "yes" or False for "no".
        """
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == "":
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


    def data_validation(self, file):
        """
        Performs data validation on the configuration file.
        Includes type checks and schema validation.
        """
        print(f'''
              Validating: configuration file...
              ''')
        config = self.load_yaml(file)
        self.check_for_exps(config)

        for i in config:
            for k, j in i.items():
                if k == 'Protocol_name':
                    if not isinstance(j, str):
                        raise ValueError(f'\'{k}\' must be of type: str')
                elif k == 'save_models':
                    if not isinstance(j, bool):
                        raise ValueError(f'\'{k}\' must be of type: bool. Eg. {k}: True')       
                elif k == 'save_datasets':
                    if not isinstance(j, bool):
                        raise ValueError(f'\'{k}\' must be of type: bool. Eg. {k}: True')
                elif k == 'save_graphs':
                    if not isinstance(j, bool):
                        raise ValueError(f'\'{k}\' must be of type: bool. Eg. {k}: True')
                elif k == 'save_nothing':
                    if not isinstance(j, bool):
                        raise ValueError(f'{k} must be of type: bool. Eg. {k}: True')
                    if j:
                        answer = self.query_yes_no(f'ATTENTION! {k} is {j}. This will permanetly delete the results folder and all results will be lost at experiment completion.\nContinue?')
                        if not answer:
                            print("Operation cancelled by user.")
                            sys.exit()
                elif k == 'uniform_initial_sample':
                    if not isinstance(j, int):
                        raise ValueError(f'\'{k}\' must be of type: int')
                elif k == 'results_path':
                    if not isinstance(j, str):
                        raise ValueError(f'\'{k}\' must be of type: str')                    
                elif k == 'unique_initial_sample':
                    schema = self.load_schema('unique_initial_sample_schema.yaml')  
                    c = Core(source_data=i, schema_data=schema)
                    c.validate(raise_exception=True)
                    if j.get('nudging') != None and len(i['unique_initial_sample']['nudging']) != 3:
                        raise ValueError("The 'nudging' list must contain exactly two elements.")
                elif k == 'retrieve_initial_sample':
                    pass
                elif k == 'Experiment':
                    schema = self.load_schema('Experiment_schema.yaml')
                    c = Core(source_data=i, schema_data=schema)
                    c.validate(raise_exception=True)

                    # Check file path for dataset files (.pkl)
                    dataset_path = j.get('dataset')
                    if dataset_path is not None and not self.check_dataset_filepath(dataset_path):
                        raise FileNotFoundError(f"Dataset file not found or not a pickle file: {dataset_path}")
                    
                    # File path for data files (.csv)                    
                    data_info = j.get('data')
                    if data_info is not None:
                        data_path = data_info.get('datafile')
                        if data_path is not None and not self.check_data_filepath(data_path):
                            raise FileNotFoundError(f"Data file not found or not a csv file: {data_path}")
                    
                    # Check that the user has provided a featurizer if they entered a data file
                    if data_info is not None:
                        featurizer_config = j.get('featurizer')
                        if featurizer_config is None:
                            raise ValueError(f"You must provide a featurizer when providing a csv file!")
                    
                elif k == 'labelExperiment':
                    schema = self.load_schema('labelExperiment_schema.yaml')
                    c = Core(source_data=i, schema_data=schema)
                    c.validate(raise_exception=True)

                    # Check file path for dataset files (.pkl)
                    dataset_path = j.get('dataset')
                    if dataset_path is not None and not self.check_dataset_filepath(dataset_path):
                        raise FileNotFoundError(f"Dataset file not found or not a pickle file: {dataset_path}")
                    
                    # File path for data files (.csv)
                    data_info = j.get('data')
                    if data_info is not None:
                        data_path = data_info.get('datafile')
                        if data_path is not None and not self.check_data_filepath(data_path):
                            raise FileNotFoundError(f"Data file not found or not a csv file: {data_path}")
                        
                    # Check that the user has provided a featurizer if they entered a data file
                    if data_info is not None:
                        featurizer_config = j.get('featurizer')
                        if featurizer_config is None:
                            raise ValueError(f"You must provide a featurizer when providing a csv file!\n"
                                             "For example\n\n"
                                             "featurizer:\n"
                                             "  name: rdkit2D\n")
                elif k == 'create_dataset':
                    schema = self.load_schema('create_dataset_schema.yaml')
                    c = Core(source_data=i, schema_data=schema)
                    c.validate(raise_exception=True)

                    # File path for data files (.csv)
                    data_info = j.get('data')
                    if data_info is not None:
                        data_path = data_info.get('datafile')
                        if data_path is not None and not self.check_data_filepath(data_path):
                            raise FileNotFoundError(f"Data file not found or not a csv file: {data_path}")
                    else:
                        raise ValueError('Please provide a .csv file.')
                        
                    # Check that the user has provided a featurizer if they entered a data file
                    if data_info is not None:
                        featurizer_config = j.get('featurizer')
                        if featurizer_config is None:
                            raise ValueError(f"You must provide a featurizer when providing a csv file!\n"
                                             "For example\n\n"
                                             "featurizer:\n"
                                             "  name: rdkit2D\n")                    
                    
                else:
                    raise ValueError(f'Error reading the configuration file at: {k}: {j}')
                
        print('''
              Data validation completed. Found no semantic errors in the configuration file.
              ''')

# v = ConfigValidator()
# v.data_validation('experiment_setups/labelExperiment.yaml')
#v.data_validation('experiment_setups/its_eksamen.yaml')