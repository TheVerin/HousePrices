#Houses Price Regression model

#importing libraries
import pandas as pd
import seaborn as sns
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
