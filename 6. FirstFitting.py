#importing libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

#Taking data to validation
from sklearn.cross_validation import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Fitting first regressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
RanReg = RandomForestRegressor()
RanReg.fit(X_train, Y_train)

GBReg = GradientBoostingRegressor()
GBReg.fit(X_train, Y_train)

import xgboost as xgb
XGBReg = xgb.XGBRegressor()
XGBReg.fit(X_train, Y_train)

#ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold

def ANNModel():
    model = Sequential()
    model.add(Dense(output_dim = 238, init = 'normal', activation = 'relu', input_dim = 238))
    model.add(Dense(output_dim = 100, init = 'normal', activation = 'relu'))
    model.add(Dense(output_dim = 1, init = 'normal'))
    model.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')
    return model

seed = 10
np.random.seed(seed)

ANNReg = KerasRegressor(build_fn = ANNModel, epochs = 100, batch_size = 5, verbose = 1)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(ANNReg, X_train, Y_train, cv=kfold)
ANNReg.fit(X_train, Y_train)


#Prediction
RanRegPred = RanReg.predict(X_val)
GBRegPred = GBReg.predict(X_val)
XGBRegPred = XGBReg.predict(X_val)
ANNRegPred = ANNReg.predict(X_val).ravel()

#Checking the RMSLE
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
RanReg_Score = rmsle(Y_val, RanRegPred)
GBReg_Score = rmsle(Y_val, GBRegPred)
XGBReg_Score = rmsle(Y_val, XGBRegPred)
ANNReg_Score = rmsle(Y_val, ANNRegPred)
print('RanRegScore = ',RanReg_Score)
print('GBRegScore = ',GBReg_Score)
print('XGBRegScore = ',XGBReg_Score)
print('ANNRegScore = ',ANNReg_Score)