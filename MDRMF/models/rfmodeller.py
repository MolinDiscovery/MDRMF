import logging
from sklearn.ensemble import RandomForestRegressor
from MDRMF.models.modeller import Modeller
from MDRMF.dataset import Dataset

class RFModeller(Modeller):

    def __init__(self, dataset, evaluator=None, iterations=10, initial_sample_size=10, acquisition_size=10, acquisition_method="greedy", retrain=True, **kwargs) -> None:
        super().__init__(dataset, evaluator, iterations, initial_sample_size, acquisition_size, acquisition_method, retrain)
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