# engine.py
import numpy as np

class Engine:
    """
    A class used to represent a Machine Learning Engine.

    Attributes
    ----------
    model : str
        The type of model to use ('RF', 'MLP', 'KNN', 'LGBM').
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
            'LGBM': self._LGBM
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

    
    def predict(self, X):
        if self.model == 'RF':
            preds = np.zeros((len(X), len(self.engine.estimators_)))
            for j, submodel in enumerate(self.engine.estimators_):
                preds[:, j] = submodel.predict(X)
            mean_preds = np.mean(preds, axis=1)
            uncertainty = np.var(preds, axis=1)

        elif self.model == 'MLP':
            mean_preds = self.engine.predict(X)
            uncertainty = None

        elif self.model == 'KNN':
            mean_preds = self.engine.predict(X)
            uncertainty = None

        elif self.model == 'LGBM':
            n_trees = self.engine.booster_.num_trees()
            preds = np.zeros((len(X), n_trees))
            for j in range(n_trees):
                preds[:, j] = self.engine.predict(X, num_iteration=j+1)
            mean_preds = np.mean(preds, axis=1)
            uncertainty = np.var(preds, axis=1)

        return mean_preds, uncertainty
    
    def _RF(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**kwargs)
    
    def _MLP(self, **kwargs):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(**kwargs)
    
    def _KNN(self, **kwargs):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(**kwargs)

    def _LGBM(self, **kwargs):
        import lightgbm as lgb
        lgb.LGBMRegressor(**kwargs)