# engine.py
import numpy as np
import os
from joblib import Parallel, delayed
num_cores = os.cpu_count()

class Engine:
    """
    A class used to represent a Machine Learning Engine.

    Attributes
    ----------
    model : str
        The type of model to use ('RF', 'MLP', 'KNN', 'LGBM', 'DT', 'SVR').
    engine : object
        The initialized machine learning model.

    Methods
    -------
    engine_start(model)
        Initializes the machine learning model based on the specified type.
    fit(X, y, **kwargs)
        Fits the model to the provided data.
    predict(X)
        Makes predictions using the fitted model and calculates uncertainty if applicable.
    """
    def __init__(self, model='RF', **kwargs) -> None:
        self.model = model
        self.engine = self.engine_select(model, **kwargs)
    
    def engine_select(self, model, **kwargs):
        engine_funcs = {
            'RF': self._RF,
            'MLP': self._MLP,
            'KNN': self._KNN,
            'LGBM': self._LGBM,
            'DT': self._DT,
            'SVR': self._SVR,
        }
        
        engine_func = engine_funcs[model]
        return engine_func(**kwargs)

    def fit(self, X, y, **kwargs):
        if self.model == 'RF':
            self.engine.fit(X, y, **kwargs)
        if self.model == 'MLP':
            self.engine.fit(X, y, **kwargs)
        if self.model == 'KNN':
            self.engine.fit(X, y, **kwargs)
        if self.model == 'LGBM':
            self.engine.fit(X, y, **kwargs)
        if self.model == 'DT':
            self.engine.fit(X, y, **kwargs)
        if self.model == 'SVR':
            self.engine.fit(X, y, **kwargs)            

    
    def predict(self, X, no_uncertainty=False):
        if self.model == 'RF':
            if no_uncertainty == False:
                preds = np.zeros((len(X), len(self.engine.estimators_)))
                for j, submodel in enumerate(self.engine.estimators_):
                    preds[:, j] = submodel.predict(X)
                mean_preds = np.mean(preds, axis=1)
                uncertainty = np.var(preds, axis=1)
            else:
                mean_preds = self.engine.predict(X)
                uncertainty = None

        elif self.model == 'MLP':
            mean_preds = self.engine.predict(X)
            uncertainty = None

        elif self.model == 'KNN':
            if no_uncertainty == False:
                neighbors_preds = self.engine.kneighbors(X, return_distance=False)
                all_preds = np.array([self.engine._y[neighbors] for neighbors in neighbors_preds])
                mean_preds = np.mean(all_preds, axis=1)
                uncertainty = np.var(all_preds, axis=1)
            else:
                mean_preds = self.engine.predict(X)
                uncertainty = None                

        elif self.model == 'LGBM':
            if no_uncertainty == False:
                n_trees = self.engine.booster_.num_trees()
                preds = np.zeros((len(X), n_trees))
                for j in range(n_trees):
                    preds[:, j] = self.engine.predict(X, num_iteration=j+1)
                mean_preds = np.mean(preds, axis=1)
                uncertainty = np.var(preds, axis=1)
            else:
                mean_preds = self.engine.predict(X)
                uncertainty = None                

        elif self.model == 'DT':
            mean_preds = self.engine.predict(X)
            uncertainty = None

        elif self.model == 'SVR':
            mean_preds = self.engine.predict(X)
            uncertainty = None

        return mean_preds, uncertainty


    def access_engine(self):
        return self.engine

    def _RF(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_jobs=-1, random_state=42, **kwargs)
    
    def _MLP(self, **kwargs):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(random_state=42, **kwargs)
    
    def _KNN(self, **kwargs):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_jobs=-1, **kwargs)

    def _LGBM(self, **kwargs):
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_jobs=-1, verbose=-1, random_state=42, **kwargs)
    
    def _DT(self, **kwargs):
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(random_state=42, **kwargs)
    
    def _SVR(self, **kwargs):
        from sklearn.svm import SVR
        return SVR(**kwargs)