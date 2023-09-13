import logging
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from MDRMF.models.modeller import Modeller
from MDRMF.dataset import Dataset

class RFModeller(Modeller):

    def __init__(
        self, 
        dataset, 
        evaluator=None, 
        iterations=10, 
        initial_sample_size=10, 
        acquisition_size=10, 
        acquisition_method="greedy", 
        retrain=True,
        seeds=[],
        **kwargs) -> None:

        super().__init__(
            dataset, 
            evaluator, 
            iterations, 
            initial_sample_size, 
            acquisition_size, 
            acquisition_method, 
            retrain,
            seeds
            )

        self.kwargs = kwargs
        self.model = RandomForestRegressor(**self.kwargs)

    def fit(self):
        
        # Get random points
        if self.seeds == []:
            initial_pts = self._initial_sampler()
        
        # If freeze_sample is not empty and it's a list of integers use this as starting points
        elif self.seeds and isinstance(self.seeds, list) and all(isinstance(i, int) for i in self.seeds):

            # Get the seeded points and remember to remove them from the dataset
            initial_pts = self.dataset.get_points(self.seeds, remove_points=True)

        else:
            logging.error("Seeds failed. Seeds must be a list of integers like [5, 25, 600, 5000]")
        
        print(f"y values of starting points {initial_pts.y}")
        self.model.fit(initial_pts.X, initial_pts.y)

        for i in range(self.iterations):
        # Acquire new points
            acquired_pts = self._acquisition(self.model)

            # Merge old and new points
            if i == 0:
                model_dataset = self.dataset.merge_datasets([initial_pts, acquired_pts])
            else:
                model_dataset = self.dataset.merge_datasets([model_dataset, acquired_pts])

            if self.retrain:
                # Reset model and train
                self.model = RandomForestRegressor(**self.kwargs)
                self.model.fit(model_dataset.X, model_dataset.y)
            else:
                # Train on existing model
                self.model.fit(model_dataset.X, model_dataset.y)

            if self.evaluator is not None:
                self.call_evaluator(i=i)

        return self.model
    
    def predict(self, dataset: Dataset):

        if isinstance(dataset, Dataset):
            return self.model.predict(dataset.X)
        else:
            logging.error("Wrong object type. Must be of type `Dataset`")

    def save(self, filename: str):
        """
        Save the RFModeller to a pickle file
        """
        # Check if filename is a string.
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
        except FileNotFoundError:
            logging.error(f"File not found: {filename}")
            raise
        except IOError as e:
            logging.error(f"IOError: {str(e)}")
            raise
        except pickle.PicklingError as e:
            logging.error(f"Failed to pickle model: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise

    @staticmethod
    def load(filename: str):
        
        # Check if filename is a string.
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        
        # Check if file exists.
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No such file or directory: '{filename}'")
        
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            logging.error(f"File not found: {filename}")
            raise
        except IOError as e:
            logging.error(f"IOError: {str(e)}")
            raise
        except pickle.UnpicklingError as e:
            logging.error(f"Failed to unpickle model: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise