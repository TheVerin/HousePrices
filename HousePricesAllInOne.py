#Houses Price Regression model

#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)


#Get the training data and first look into variables
RawData = pd.read_csv('train.csv')

#Chcecking the data
RawData.describe()
RawData.dtypes
RawData.hist(bins = 30)

#looking for nan values
nulls_summary = pd.DataFrame(RawData.isnull().any(), columns = ['Nulls'])
nulls_summary['Number of nans'] = pd.DataFrame(RawData.isnull().sum())
nulls_summary['Percentage of nans'] = round((RawData.isnull().mean()*100),2)
print(nulls_summary)

#checking how namy categorical values is in each column
for col in RawData.select_dtypes(['object']):
    print(RawData[col].value_counts())
    
#time for some visualisation
sns.stripplot(data = RawData, x = 'SalePrice', y = 'SalePrice')
sns.stripplot(data = RawData, y = 'GarageYrBlt', x = 'YearRemodAdd')
sns.stripplot(data = RawData, y = 'SalePrice', x = 'LotArea')
sns.stripplot(data = RawData, y = 'SalePrice', x = 'MSSubClass')
sns.boxplot(data = RawData, x = 'YearBuilt' , y = 'SalePrice')
sns.boxplot(data = RawData, x = 'YearRemodAdd' , y = 'SalePrice')
sns.boxplot(data = RawData, x = 'OverallQual', y = 'SalePrice')
sns.boxplot(data = RawData, x = 'OverallCond', y = 'SalePrice')
sns.boxplot(data = RawData, x = 'Neighborhood', y = 'SalePrice')
sns.boxplot(data = RawData, x = 'YrSold', y = 'SalePrice')
sns.boxplot(data = RawData, x = 'SaleCondition', y = 'SalePrice')
sns.boxplot(data = RawData, x = 'BsmtQual', y = 'SalePrice')
sns.boxplot(data = RawData, x = 'GarageCond', y = 'SalePrice')
sns.boxplot(data = RawData, x = 'GarageType', y = 'SalePrice')
sns.boxplot(data = RawData, x = 'MSZoning', y = 'SalePrice')
sns.stripplot(data = RawData, x = 'LotFrontage', y = 'SalePrice')

#get combined Data
def combined_data():
    train = pd.read_csv('train.csv')
    train.drop('SalePrice', axis = 1, inplace = True)
    
    test = pd.read_csv('test.csv')
    AllData = train.append(test)
    AllData.reset_index(inplace = True)
    AllData.drop(['index', 'Id'], axis = 1, inplace = True)
    
    return AllData
AllData = combined_data()

#Getting target data
Y = RawData.iloc[:, -1]

#Checking for NaN values in all columns
nulls_summary = pd.DataFrame(AllData.isnull().any(), columns = ['Nulls'])
nulls_summary['Number of nans'] = pd.DataFrame(AllData.isnull().sum())
nulls_summary['Percentage of nans'] = round((AllData.isnull().mean()*100),2)
print(nulls_summary)

#Getting away columns with Nan precentage ~50%
AllData.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'Street', 'Utilities'], axis = 1, inplace = True)


#Dealing with 0-1 values
#Paved drive
def PavedDrive():
    global AllData
    AllData['PavedDrive'] = AllData['PavedDrive'].map({'Y':1, 'N':0})
    return AllData
AllData = PavedDrive()

#Central Air
def CentralAir():
    global AllData
    AllData['CentralAir'] = AllData['CentralAir'].map({'Y':1, 'N':0})
    return AllData
AllData = CentralAir()

def HeatingQC():
    global AllData
    AllData['HeatingQC'] = AllData['HeatingQC'].map({'Y':1, 'N':0})
    return AllData
AllData = HeatingQC()


#Filling NaN vaues
#At first dealing with the strings values
AllData.GarageFinish.fillna(value = 'No', inplace = True)
AllData.GarageQual.fillna(value = 'No', inplace = True)
AllData.GarageCond.fillna(value = 'No', inplace = True)
AllData.GarageType.fillna(value = 'No', inplace = True)
AllData.BsmtCond.fillna(value = 'No', inplace = True)
AllData.BsmtExposure.fillna(value = 'No', inplace = True)
AllData.BsmtQual.fillna(value = 'No', inplace = True)
AllData.BsmtFinType2.fillna(value = 'No', inplace = True)
AllData.BsmtFinType1.fillna(value = 'No', inplace = True)
AllData.MasVnrType.fillna(value = 'No', inplace = True)

AllData.MSZoning.fillna(value = 'RL', inplace = True)
AllData.Functional.fillna(value = 'Typ', inplace = True)
AllData.Exterior1st.fillna(value = 'VinylSd', inplace = True)
AllData.Exterior2nd.fillna(value = 'VinylSd', inplace = True)
AllData.Electrical.fillna(value = 'SBrkr', inplace = True)
AllData.KitchenQual.fillna(value = 'TA', inplace = True)
AllData.SaleType.fillna(value = 'WD', inplace = True)

#Now inplacementing '0'
AllData.MasVnrArea.fillna(value = 0, inplace = True)
AllData.BsmtFinSF1.fillna(value = 0, inplace = True)
AllData.BsmtFinSF2.fillna(value = 0, inplace = True)
AllData.BsmtUnfSF.fillna(value = 0, inplace = True)
AllData.TotalBsmtSF.fillna(value = 0, inplace = True)

#Fillin with int values
AllData.LotFrontage.fillna(value = AllData['LotFrontage'].median(), inplace = True)
AllData.GarageYrBlt.fillna(value = AllData['YearBuilt'], inplace = True)
AllData.BsmtFullBath.fillna(value = AllData['BsmtFullBath'].median(), inplace = True)
AllData.BsmtHalfBath.fillna(value = AllData['BsmtHalfBath'].median(), inplace = True)
AllData.GarageCars.fillna(value = AllData['GarageCars'].median(), inplace = True)
AllData.GarageArea.fillna(value = AllData['GarageArea'].median(), inplace = True)
AllData.PavedDrive.fillna(value = AllData['PavedDrive'].median(), inplace = True)


#Getting dummies
def FunctionalDummies():
    global AllData
    functional_dummies = pd.get_dummies(AllData['Functional'], prefix = 'Functional')
    AllData = pd.concat([AllData, functional_dummies], axis = 1)
    AllData.drop('Functional', axis = 1, inplace = True)
    return AllData
AllData = FunctionalDummies()
AllData = AllData.iloc[:, :-1]

def KitchenQualDummies():
    global AllData
    KitchenQual_dummies = pd.get_dummies(AllData['KitchenQual'], prefix = 'KitchenQual')
    AllData = pd.concat([AllData, KitchenQual_dummies], axis = 1)
    AllData.drop('KitchenQual', axis = 1, inplace = True)
    return AllData
AllData = KitchenQualDummies()
AllData = AllData.iloc[:, :-1]

def MSZoningDummies():
    global AllData
    MSZoning_dummies = pd.get_dummies(AllData['MSZoning'], prefix = 'MSZoning')
    AllData = pd.concat([AllData, MSZoning_dummies], axis = 1)
    AllData.drop('MSZoning', axis = 1, inplace = True)
    return AllData
AllData = MSZoningDummies()
AllData = AllData.iloc[:, :-1]

def LotShapeDummies():
    global AllData
    LotShape_dummies = pd.get_dummies(AllData['LotShape'], prefix = 'LotShape')
    AllData = pd.concat([AllData, LotShape_dummies], axis = 1)
    AllData.drop('LotShape', axis = 1, inplace = True)
    return AllData
AllData = LotShapeDummies()
AllData = AllData.iloc[:, :-1]

def LandContourDummies():
    global AllData
    LandContour_dummies = pd.get_dummies(AllData['LandContour'], prefix = 'LandContour')
    AllData = pd.concat([AllData, LandContour_dummies], axis = 1)
    AllData.drop('LandContour', axis = 1, inplace = True)
    return AllData
AllData = LandContourDummies()
AllData = AllData.iloc[:, :-1]

def LotConfigDummies():
    global AllData
    LotConfig_dummies = pd.get_dummies(AllData['LotConfig'], prefix = 'LotConfig')
    AllData = pd.concat([AllData, LotConfig_dummies], axis = 1)
    AllData.drop('LotConfig', axis = 1, inplace = True)
    return AllData
AllData = LotConfigDummies()
AllData = AllData.iloc[:, :-1]

def LandSlopeDummies():
    global AllData
    LandSlope_dummies = pd.get_dummies(AllData['LandSlope'], prefix = 'LandSlope')
    AllData = pd.concat([AllData, LandSlope_dummies], axis = 1)
    AllData.drop('LandSlope', axis = 1, inplace = True)
    return AllData
AllData = LandSlopeDummies()
AllData = AllData.iloc[:, :-1]

def NeighborhoodDummies():
    global AllData
    Neighborhood_dummies = pd.get_dummies(AllData['Neighborhood'], prefix = 'Neighborhood')
    AllData = pd.concat([AllData, Neighborhood_dummies], axis = 1)
    AllData.drop('Neighborhood', axis = 1, inplace = True)
    return AllData
AllData = NeighborhoodDummies()
AllData = AllData.iloc[:, :-1]

def Condition1Dummies():
    global AllData
    Condition1_dummies = pd.get_dummies(AllData['Condition1'], prefix = 'Condition1')
    AllData = pd.concat([AllData, Condition1_dummies], axis = 1)
    AllData.drop('Condition1', axis = 1, inplace = True)
    return AllData
AllData = Condition1Dummies()
AllData = AllData.iloc[:, :-1]

def Condition2Dummies():
    global AllData
    Condition2_dummies = pd.get_dummies(AllData['Condition2'], prefix = 'Condition2')
    AllData = pd.concat([AllData, Condition2_dummies], axis = 1)
    AllData.drop('Condition2', axis = 1, inplace = True)
    return AllData
AllData = Condition2Dummies()
AllData = AllData.iloc[:, :-1]

def BldgTypeDummies():
    global AllData
    BldgType_dummies = pd.get_dummies(AllData['BldgType'], prefix = 'BldgType')
    AllData = pd.concat([AllData, BldgType_dummies], axis = 1)
    AllData.drop('BldgType', axis = 1, inplace = True)
    return AllData
AllData = BldgTypeDummies()
AllData = AllData.iloc[:, :-1]

def HouseStyleDummies():
    global AllData
    HouseStyle_dummies = pd.get_dummies(AllData['HouseStyle'], prefix = 'HouseStyle')
    AllData = pd.concat([AllData, HouseStyle_dummies], axis = 1)
    AllData.drop('HouseStyle', axis = 1, inplace = True)
    return AllData
AllData = HouseStyleDummies()
AllData = AllData.iloc[:, :-1]

def RoofStyleDummies():
    global AllData
    RoofStyle_dummies = pd.get_dummies(AllData['RoofStyle'], prefix = 'RoofStyle')
    AllData = pd.concat([AllData, RoofStyle_dummies], axis = 1)
    AllData.drop('RoofStyle', axis = 1, inplace = True)
    return AllData
AllData = RoofStyleDummies()
AllData = AllData.iloc[:, :-1]

def RoofMatlDummies():
    global AllData
    RoofMatl_dummies = pd.get_dummies(AllData['RoofMatl'], prefix = 'RoofMatl')
    AllData = pd.concat([AllData, RoofMatl_dummies], axis = 1)
    AllData.drop('RoofMatl', axis = 1, inplace = True)
    return AllData
AllData = RoofMatlDummies()
AllData = AllData.iloc[:, :-1]

def Exterior1stDummies():
    global AllData
    Exterior1st_dummies = pd.get_dummies(AllData['Exterior1st'], prefix = 'Exterior1st')
    AllData = pd.concat([AllData, Exterior1st_dummies], axis = 1)
    AllData.drop('Exterior1st', axis = 1, inplace = True)
    return AllData
AllData = Exterior1stDummies()
AllData = AllData.iloc[:, :-1]

def Exterior2ndDummies():
    global AllData
    Exterior2nd_dummies = pd.get_dummies(AllData['Exterior2nd'], prefix = 'Exterior2nd')
    AllData = pd.concat([AllData, Exterior2nd_dummies], axis = 1)
    AllData.drop('Exterior2nd', axis = 1, inplace = True)
    return AllData
AllData = Exterior2ndDummies()
AllData = AllData.iloc[:, :-1]

def MasVnrTypeDummies():
    global AllData
    MasVnrType_dummies = pd.get_dummies(AllData['MasVnrType'], prefix = 'MasVnrType')
    AllData = pd.concat([AllData, MasVnrType_dummies], axis = 1)
    AllData.drop('MasVnrType', axis = 1, inplace = True)
    return AllData
AllData = MasVnrTypeDummies()
AllData = AllData.iloc[:, :-1]

def ExterQualDummies():
    global AllData
    ExterQual_dummies = pd.get_dummies(AllData['ExterQual'], prefix = 'ExterQual')
    AllData = pd.concat([AllData, ExterQual_dummies], axis = 1)
    AllData.drop('ExterQual', axis = 1, inplace = True)
    return AllData
AllData = ExterQualDummies()
AllData = AllData.iloc[:, :-1]

def ExterCondDummies():
    global AllData
    ExterCond_dummies = pd.get_dummies(AllData['ExterCond'], prefix = 'ExterCond')
    AllData = pd.concat([AllData, ExterCond_dummies], axis = 1)
    AllData.drop('ExterCond', axis = 1, inplace = True)
    return AllData
AllData = ExterCondDummies()
AllData = AllData.iloc[:, :-1]

def FoundationDummies():
    global AllData
    Foundation_dummies = pd.get_dummies(AllData['Foundation'], prefix = 'Foundation')
    AllData = pd.concat([AllData, Foundation_dummies], axis = 1)
    AllData.drop('Foundation', axis = 1, inplace = True)
    return AllData
AllData = FoundationDummies()
AllData = AllData.iloc[:, :-1]

def BsmtQualDummies():
    global AllData
    BsmtQual_dummies = pd.get_dummies(AllData['BsmtQual'], prefix = 'BsmtQual')
    AllData = pd.concat([AllData, BsmtQual_dummies], axis = 1)
    AllData.drop('BsmtQual', axis = 1, inplace = True)
    return AllData
AllData = BsmtQualDummies()
AllData = AllData.iloc[:, :-1]

def BsmtCondDummies():
    global AllData
    BsmtCond_dummies = pd.get_dummies(AllData['BsmtCond'], prefix = 'BsmtCond')
    AllData = pd.concat([AllData, BsmtCond_dummies], axis = 1)
    AllData.drop('BsmtCond', axis = 1, inplace = True)
    return AllData
AllData = BsmtCondDummies()
AllData = AllData.iloc[:, :-1]

def BsmtExposureDummies():
    global AllData
    BsmtExposure_dummies = pd.get_dummies(AllData['BsmtExposure'], prefix = 'BsmtExposure')
    AllData = pd.concat([AllData, BsmtExposure_dummies], axis = 1)
    AllData.drop('BsmtExposure', axis = 1, inplace = True)
    return AllData
AllData = BsmtExposureDummies()
AllData = AllData.iloc[:, :-1]

def BsmtFinType1Dummies():
    global AllData
    BsmtFinType1_dummies = pd.get_dummies(AllData['BsmtFinType1'], prefix = 'BsmtFinType1')
    AllData = pd.concat([AllData, BsmtFinType1_dummies], axis = 1)
    AllData.drop('BsmtFinType1', axis = 1, inplace = True)
    return AllData
AllData = BsmtFinType1Dummies()
AllData = AllData.iloc[:, :-1]

def BsmtFinType2Dummies():
    global AllData
    BsmtFinType2_dummies = pd.get_dummies(AllData['BsmtFinType2'], prefix = 'BsmtFinType2')
    AllData = pd.concat([AllData, BsmtFinType2_dummies], axis = 1)
    AllData.drop('BsmtFinType2', axis = 1, inplace = True)
    return AllData
AllData = BsmtFinType2Dummies()
AllData = AllData.iloc[:, :-1]

def HeatingDummies():
    global AllData
    Heating_dummies = pd.get_dummies(AllData['Heating'], prefix = 'Heating')
    AllData = pd.concat([AllData, Heating_dummies], axis = 1)
    AllData.drop('Heating', axis = 1, inplace = True)
    return AllData
AllData = HeatingDummies()
AllData = AllData.iloc[:, :-1]

def ElectricalDummies():
    global AllData
    Electrical_dummies = pd.get_dummies(AllData['Electrical'], prefix = 'Electrical')
    AllData = pd.concat([AllData, Electrical_dummies], axis = 1)
    AllData.drop('Electrical', axis = 1, inplace = True)
    return AllData
AllData = ElectricalDummies()
AllData = AllData.iloc[:, :-1]

def GarageTypeDummies():
    global AllData
    GarageType_dummies = pd.get_dummies(AllData['GarageType'], prefix = 'GarageType')
    AllData = pd.concat([AllData, GarageType_dummies], axis = 1)
    AllData.drop('GarageType', axis = 1, inplace = True)
    return AllData
AllData = GarageTypeDummies()
AllData = AllData.iloc[:, :-1]

def GarageFinishDummies():
    global AllData
    GarageFinish_dummies = pd.get_dummies(AllData['GarageFinish'], prefix = 'GarageFinish')
    AllData = pd.concat([AllData, GarageFinish_dummies], axis = 1)
    AllData.drop('GarageFinish', axis = 1, inplace = True)
    return AllData
AllData = GarageFinishDummies()
AllData = AllData.iloc[:, :-1]

def GarageQualDummies():
    global AllData
    GarageQual_dummies = pd.get_dummies(AllData['GarageQual'], prefix = 'GarageQual')
    AllData = pd.concat([AllData, GarageQual_dummies], axis = 1)
    AllData.drop('GarageQual', axis = 1, inplace = True)
    return AllData
AllData = GarageQualDummies()
AllData = AllData.iloc[:, :-1]

def GarageCondDummies():
    global AllData
    GarageCond_dummies = pd.get_dummies(AllData['GarageCond'], prefix = 'GarageCond')
    AllData = pd.concat([AllData, GarageCond_dummies], axis = 1)
    AllData.drop('GarageCond', axis = 1, inplace = True)
    return AllData
AllData = GarageCondDummies()
AllData = AllData.iloc[:, :-1]

def SaleConditionDummies():
    global AllData
    SaleCondition_dummies = pd.get_dummies(AllData['SaleCondition'], prefix = 'SaleCondition')
    AllData = pd.concat([AllData, SaleCondition_dummies], axis = 1)
    AllData.drop('SaleCondition', axis = 1, inplace = True)
    return AllData
AllData = SaleConditionDummies()
AllData = AllData.iloc[:, :-1]

def SaleTypeDummies():
    global AllData
    SaleType_dummies = pd.get_dummies(AllData['SaleType'], prefix = 'SaleType')
    AllData = pd.concat([AllData, SaleType_dummies], axis = 1)
    AllData.drop('SaleType', axis = 1, inplace = True)
    return AllData
AllData = SaleTypeDummies()
AllData = AllData.iloc[:, :-1]

def HeatingQCDummies():
    global AllData
    HeatingQC_dummies = pd.get_dummies(AllData['HeatingQC'], prefix = 'HeatingQC')
    AllData = pd.concat([AllData, HeatingQC_dummies], axis = 1)
    AllData.drop('HeatingQC', axis = 1, inplace = True)
    return AllData
AllData = HeatingQCDummies()
AllData = AllData.iloc[:, :-1]


#Splitting for test and train set
X = AllData.iloc[:1460, :]
TestData = AllData.iloc[1460:, :]


#Scalling without reduction
#Scalling variables
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X_S = StandardScaler()
X = X_S.fit_transform(X)
TestData = X_S.fit_transform(TestData)

MM = MinMaxScaler(feature_range = (0,1))
X = MM.fit_transform(X)
TestData = MM.fit_transform(TestData)

#Comming back to dataframe
X = pd.DataFrame(X)
TestData = pd.DataFrame(TestData)

#Taking data to validation
from sklearn.cross_validation import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Fitting first regressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
RanReg = RandomForestRegressor()
RanReg.fit(X, Y)

GBReg = GradientBoostingRegressor()
GBReg.fit(X, Y)

import xgboost as xgb
XGBReg = xgb.XGBRegressor()
XGBReg.fit(X, Y)

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
RanRegPred = RanReg.predict(X)
GBRegPred = GBReg.predict(X)
XGBRegPred = XGBReg.predict(X)
ANNRegPred = ANNReg.predict(X_val).ravel()

#Checking the RMSLE
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
RanReg_Score = rmsle(Y, RanRegPred)
GBReg_Score = rmsle(Y, GBRegPred)
XGBReg_Score = rmsle(Y, XGBRegPred)
ANNReg_Score = rmsle(Y_train, ANNRegPred)
print('RanRegScore = ',RanReg_Score)
print('GBRegScore = ',GBReg_Score)
print('XGBRegScore = ',XGBReg_Score)
print('ANNRegScore = ',ANNReg_Score)

#chgecking features -> only for dataframes
features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = GBReg.feature_importances_
features.sort_values(by = 'importance', ascending = True, inplace = True)
features.set_index('feature', inplace = True)
features.plot(kind = 'barh')

#Taking only important features
from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(GBReg, prefit = True)
X_reduced = model.transform(X)
X_train_reduced = model.transform(X_train)
X_val_reduced = model.transform(X_val)
TestData_reduced = model.transform(TestData)
print(X_train_reduced.shape)

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

df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['Id'] = aux['Id']
df_output['SalePrice'] = Prediction
df_output[['Id','SalePrice']].to_csv('GBReg_best.csv', index=False)
