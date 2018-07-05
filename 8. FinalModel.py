#importing libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

#Next Fitting on a reduced dataset on the best alghorytm
GBReg_reduced = GradientBoostingRegressor()
GBReg_reduced.fit(X_reduced, Y)

#Prediction
GBRegPred_reduced = GBReg_reduced.predict(X_reduced)

#Checking the RMSLE
GBReg_reduced_Score = rmsle(Y, GBRegPred_reduced)
print('RanRegScore = ',GBReg_reduced_Score)

#Taking GridSearch for it
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':np.arange(100, 500, 100),
              'max_features':['sqrt', 'auto', 'log2'],
              'max_depth':np.arange(1,8),
              'min_samples_split':np.arange(2,5),
              'min_samples_leaf':np.arange(1,4),
              'learning_rate':[0.05, 0.1, 0.5, 1]}
grid_search = GridSearchCV(GBReg_reduced, scoring = 'neg_mean_squared_log_error', param_grid = parameters, cv = 10, verbose = 1)
grid_search.fit(X_reduced, Y)
bests = grid_search.best_params_
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



#Making prediction for test data (with the best algorithm)
FinishModel = GradientBoostingRegressor(n_estimators = 100)
FinishModel.fit(X, Y)
Prediction = FinishModel.predict(TestData)