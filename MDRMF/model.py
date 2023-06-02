from MDRMF import Dataset
from MDRMF.models import Modeller

class Model:
    def __init__(self, model: Modeller) -> None:
        self.model = model

    def train(self):
        self.model.fit()

    def predict(self, dataset: Dataset):
        return self.model.predict(dataset)
    
    @property
    def results(self):
        return self.model.results