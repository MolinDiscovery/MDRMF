import yaml
from pykwalify.core import Core
from MDRMF import Dataset


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
    

    def load_schema(self, file):
        """Load and return the YAML schema from the specified file."""
        with open(file, 'r') as file:
            return yaml.safe_load(file)


    def check_for_exps(self, config):
        """
        Validates the config for exclusive presence of 'Experiment' or 'labelExperiment'.
        Raises exceptions for invalid or conflicting configurations.
        """
        e_set = set()
        for i in config:
            if not isinstance(i, dict):
                raise ValueError(f'This top-level key is not accepted: {i}')
            
            for j in i.keys():
                e_set.add(j)

        if 'Experiment' in e_set and 'labelExperiment' in e_set:
            raise Exception('You cannot conduct "Experiment" and "labelExperiment" at the same time!')
        elif 'Experiment' in e_set or 'labelExperiment' in e_set:
            pass
        else:
            raise Exception('''
    Fatal error while reading the config file.
    Please, include only one "Experiment" or "labelExperiment",
    and check structure of your config file.
                            ''')


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
                elif k == 'save_nothing':
                    if not isinstance(j, bool):
                        raise ValueError(f'{k} must be of type: bool. Eg. {k}: True')
                elif k == 'unique_initial_sample':
                    schema = self.load_schema('MDRMF/schemas/unique_initial_sample_schema.yaml')  
                    c = Core(source_data=i, schema_data=schema)
                    c.validate(raise_exception=True)
                    if j.get('nudging') != None and len(i['unique_initial_sample']['nudging']) != 2:
                        raise ValueError("The 'nudging' list must contain exactly two elements.")
                elif k == 'Experiment':
                    schema = self.load_schema('MDRMF/schemas/Experiment_schema.yaml')
                    c = Core(source_data=i, schema_data=schema)
                    c.validate(raise_exception=True)
                    if j.get('dataset') is not None:
                        pass
                        Dataset.load(j.get('dataset')) # Preemptively loads every dataset to see if any of them fails.
                elif k == 'labelExperiment':
                    schema = self.load_schema('MDRMF/schemas/labelExperiment_schema.yaml')
                    c = Core(source_data=i, schema_data=schema)
                    c.validate(raise_exception=True)
                    if j.get('dataset') is not None:
                        pass
                        Dataset.load(j.get('dataset')) # Preemptively loads every dataset to see if any of them fails.
                else:
                    raise ValueError(f'This top-level key is not accepted in the settings: {k}')
                
        print('''
              Data validation completed. Found no semantic errors in the configuration file.
              ''')

v = ConfigValidator()
v.data_validation('experiment_setups/pairwise_vs_descriptor.yaml')