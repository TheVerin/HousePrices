#importing libraries
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

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
