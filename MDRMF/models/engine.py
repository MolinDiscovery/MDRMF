# engine.py
import numpy as np

class Engine:

    def __init__(
            self, 
            model='RF',
            **kwargs,
            ) -> None:
        
        self.model
        self.engine = self.engine_start(model)
        self.engine(**kwargs)
    
    def engine_start(self, model):
        engine_funcs = {
            'RF': self._RF,
            'MLP': self._MLP,
            'KNN': self._KNN,
            'LGBM': self._LGBM
        }
        
        engine = engine_funcs[self.model]
        return engine

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
                preds[:,j] = submodel.predict(X)
            preds = np.mean(preds, axis=1)
            uncertainty = np.var(preds, axis=1)

        if self.model == 'MLP':
            preds = self.engine.predict(X)
            uncertainty = None

        if self.model == 'KNN':
            preds = self.engine.predict(X)
            uncertainty = None

        if self.model == 'LGBM':
            n_trees = self.engine.num_trees()
            preds = np.zeros((len(X), n_trees))
            for j in range(n_trees):
                preds[:, j] = self.engine.predict(X, num_iteration=j+1, pred_leaf=False, raw_score=False)
            preds = np.mean(preds,axis=1)
            uncertainty = np.var(preds, axis=1)
        return preds, uncertainty


def pred_with_uncertainty(X, model, n_trees=None):
    # If n_trees is not specified, use all trees
    if n_trees is None:
        n_trees = model.num_trees()

    # Initialize an array to hold predictions from each tree
    preds = np.zeros((len(X), n_trees))

    # Get predictions from each tree
    for j in range(n_trees):
        preds[:, j] = model.predict(X, num_iteration=j+1, pred_leaf=False, raw_score=False)

    # Calculate the mean and variance across trees
    return np.mean(preds, axis=1), np.var(preds, axis=1)        
    
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

    

model = Engine('RF', n_estimators=100)