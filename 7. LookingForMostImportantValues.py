#importing libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

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