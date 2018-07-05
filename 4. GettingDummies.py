#importing libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category = DeprecationWarning)

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