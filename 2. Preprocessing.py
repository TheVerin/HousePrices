#importing libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

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