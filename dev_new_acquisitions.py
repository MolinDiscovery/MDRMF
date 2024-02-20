from MDRMF import Dataset
from MDRMF.models import RFModeller

dataset = Dataset.load('dataset.pkl')
model = RFModeller(dataset, acquisition_method='LCB')
model.fit()