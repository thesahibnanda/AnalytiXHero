# Import Required Libraries
import warnings

import numpy as np
import pandas as pd

from _basic_functions import ConvertDataFrame, DataFrameToNumPy

__all__ = ['TreatNull']

# Now We'll Define Functions That Will Contribute For Our Main Feature

# Function 1, Replace Empty Strings Or Blank Spaces With NaN
def ReplaceEmpty(DataFrame):
    '''
        Args
        ----
        DataFrame: Pandas DataFrame, The Dataset Given By User

        Returns
        -------
        DataFrame After Treating Empty Spaces
    '''
    DataFrame.replace(["", " ", "  ", "   ", "    ", "     "], np.nan, inplace=True)

    return DataFrame

# Function 2, Drop Null Spaces
def Remove_Null(df):
    '''
        Args
        ----
        df: Pandas DataFrame, The Dataset Given By User

        Returns
        -------
        df: DataFrame After Deleting Null Spaces
    '''
    df = df.dropna(axis=0)
    return df

def Impute_Null(df, method):
    '''
        Args
        ----
        df: Pandas DataFrame Given By User

        method: 'str' User Specifies Which Value Will Be Used For Imputation (Possible Values = ['mean', 'median', 'mode', 'std'])
    '''
    imputed_df = df.copy()
    for column in imputed_df.columns:
        if imputed_df[column].isnull().any():
            column_dtype = imputed_df[column].dtype
            
            if column_dtype == object or column_dtype == 'datetime64[ns]' or column_dtype == bool:
                warnings.warn(f"Column '{column}' has object or datetime or boolean data type. Imputing NaN values with mode.")
                imputed_df[column].fillna(imputed_df[column].mode().iloc[0], inplace=True)
            else:
                if method == 'mean':
                    imputed_df[column].fillna(imputed_df[column].mean(), inplace=True)
                elif method == 'median':
                    imputed_df[column].fillna(imputed_df[column].median(), inplace=True)
                elif method == 'mode':
                    imputed_df[column].fillna(imputed_df[column].mode().iloc[0], inplace=True)
                elif method == 'std':
                    imputed_df[column].fillna(imputed_df[column].std(), inplace=True)
                else:
                    raise ValueError("Invalid method. Possible values are ['mean', 'median', 'mode', 'std'].")
    return imputed_df

# These Will Be Our Main Features

class TreatNull:
    '''
        Used To Treat Null Spaces, Datapoints Containing Null Spaces Either Can Be Imputed Or Removed
    
        Parameters
        ----------
            treatment: 'str', Optional (Default = 'impute')
            User Specifies What Treatment Wil Outliers Undergo (Possible Values = ['impute','remove'])

            imputation: 'str', Optional (Default = 'mean')
            User Specifies Which Value Will Be Used For Imputation (Possible Values = ['mean', 'median', 'mode', 'std'])
            Only Useful If treatment = 'impute'

            index_present: 'bool', Optional (Default = False)
            User Specifies That If The First Column Is Index Or Some Important Column

            target_split: 'bool', Optional (Default = False)
            User Specifies Whether The Last Column Of The Dataset Is A Target Column And Whether User Wants It To Be Returned As A Different Variable
        
        Returns
        -------
            numpy.ndarray After Outlier Treatment
    '''

    def __init__(self,
                treatment = 'impute',
                imputation = 'mean',
                index_present = False,
                target_split = False):
        
        if treatment not in ['remove', 'impute']:
            raise ValueError("treatment should be either 'impute' or 'remove'")
        
        if imputation not in ['mean', 'median', 'mode', 'std']:
            raise ValueError("imputation should be in ['mean', 'median', 'mode', 'std']")
        
        if index_present not in [True, False]:
            raise ValueError('index_present should be a <bool>')
        
        if target_split not in [True, False]:
            raise ValueError('target_split should be a <bool>')
        
        if treatment == "remove" and imputation != 'mean':
            warnings.warn("No Need Of Mentioning Imputation As Null Spaces Will Be Removed")
        
        self.treatment = treatment
        self.imputation = imputation
        self.index_present = index_present
        self.target_split = target_split

    
    def fit_transform(self, Dataset):
        '''
            Parameters
            ----------
                Dataset: This Dataset Will Undergo Null Values Treatment
                Any Supported Data Structure (List, Tuple, Dictionary, Numpy.Ndarray, Pandas.DataFrame)
        '''

        Dataset_or = ConvertDataFrame(Dataset, self.index_present)[0]

        Dataset = Dataset_or.copy()

        Dataset = ReplaceEmpty(Dataset)

        if self.treatment == 'remove':
            Dataset = Remove_Null(Dataset)
        elif self.treatment == 'impute':
            Dataset = Impute_Null(Dataset, self.imputation)
        
        if self.target_split == True:
            X = pd.DataFrame(Dataset.iloc[:, :-1])
            y = pd.DataFrame(Dataset.iloc[:, -1])
            
            X = DataFrameToNumPy(X)
            y = DataFrameToNumPy(y)

            return X, y
        else:
            Dataset = pd.DataFrame(Dataset)
            Dataset  = DataFrameToNumPy(Dataset)
        
            return Dataset
    
