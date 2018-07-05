#importing libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

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