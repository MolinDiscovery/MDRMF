import logging
from sklearn.ensemble import RandomForestRegressor
from .modeller import Modeller
from ..dataset import Dataset

class RFModeller(Modeller):

    def __init__(self, dataset, iterations=10, initial_sample_size=10, acquisition_size=10, acquisition_method="greedy", retrain=True, **kwargs) -> None:
        super().__init__(dataset, iterations, initial_sample_size, acquisition_size, acquisition_method, retrain)
        from sklearn.ensemble import RandomForestRegressor
        self.kwargs = kwargs
        self.model = RandomForestRegressor(**self.kwargs)

    def fit(self):
        
        # Get random points
        random_pts = self._initial_sampler()

        # Fit initial model
        #for i in range(self.iterations):
            
        self.model.fit(random_pts.X, random_pts.y)

        for i in range(self.iterations):
        # Acquire new points
            acquired_pts = self._acquisition(self.model)

            # Merge old and new points
            if i == 0:
                model_dataset = self.dataset.merge_datasets([random_pts, acquired_pts])
            else:
                model_dataset = self.dataset.merge_datasets([model_dataset, acquired_pts])

            if self.retrain:
                # Retrain with new model_dataset
                self.model = RandomForestRegressor(**self.kwargs)
                self.model.fit(model_dataset.X, model_dataset.y)
            else:
                # Train on existing model
                self.model.fit(model_dataset.X, model_dataset.y)


        return self.model #model_dataset
    
    def predict(self, dataset: Dataset):

        if isinstance(dataset, Dataset):
            return self.model.predict(dataset.X)
        else:
            logging.error("Wrong object type. Must be of type `Dataset`")