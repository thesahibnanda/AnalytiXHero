# Import Required Libraries
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox

from _basic_functions import ConvertDataFrame, DataFrameToNumPy

__all__ = ["Transform"]

# Replacement Function
def replace_nan_inf(dataframe_or):
    dataframe = dataframe_or.copy()
    dataframe.replace(np.nan, 0, inplace=True)
    dataframe.replace(np.inf, 2**31 - 1, inplace=True)
    dataframe.replace(-np.inf, -2**31 + 1, inplace=True) 
    return dataframe

# All Transformation Functions That Will Contribute In Main Features
def log_transform(dataframe, column_name, delete_old):
    transformed_df = dataframe.copy()
    transformed_col = np.log(dataframe[column_name])
    transformed_df.insert(dataframe.columns.get_loc(column_name) + 1, f"{column_name}_log", transformed_col)
    if delete_old:
        transformed_df.drop(column_name, axis=1, inplace=True)
    return transformed_df

def log1p_transform(dataframe, column_name, delete_old):
    transformed_df = dataframe.copy()
    transformed_col = np.log1p(dataframe[column_name])
    transformed_df.insert(dataframe.columns.get_loc(column_name) + 1, f"{column_name}_log1p", transformed_col)
    if delete_old:
        transformed_df.drop(column_name, axis=1, inplace=True)
    return transformed_df

def log_plus_1_transform(dataframe, column_name, delete_old):
    transformed_df = dataframe.copy()
    transformed_col = 1 + np.log(dataframe[column_name])
    transformed_df.insert(dataframe.columns.get_loc(column_name) + 1, f"{column_name}_1+log", transformed_col)
    if delete_old:
        transformed_df.drop(column_name, axis=1, inplace=True)
    return transformed_df


def boxcox_transform(dataframe, column_name, delete_old=False):
    transformed_df = dataframe.copy()
    data = dataframe[column_name]
    
    if data.min() <= 0:
        warnings.warn(f"Column '{column_name}' contains non-positive values. Adding a small constant to make the data positive for boxcox transformation.")
        # Adding a small constant to make the data positive
        data = data - data.min() + 1e-8
    
    transformed_col, _ = boxcox(data)
    transformed_df.insert(dataframe.columns.get_loc(column_name) + 1, f"{column_name}_boxcox", transformed_col)
    
    if delete_old:
        transformed_df.drop(column_name, axis=1, inplace=True)
    
    return transformed_df

def root_transform(dataframe, column_name, root=2, delete_old=False):
    transformed_df = dataframe.copy()
    transformed_col = np.power(dataframe[column_name], 1 / root)
    transformed_df.insert(dataframe.columns.get_loc(column_name) + 1, f"{column_name}_root_{root}", transformed_col)
    if delete_old:
        transformed_df.drop(column_name, axis=1, inplace=True)
    return transformed_df

def exp_transform(dataframe, column_name, delete_old):
    transformed_df = dataframe.copy()
    transformed_col = np.exp(dataframe[column_name])
    transformed_df.insert(dataframe.columns.get_loc(column_name) + 1, f"{column_name}_exp", transformed_col)
    if delete_old:
        transformed_df.drop(column_name, axis=1, inplace=True)
    return transformed_df

def inverse_transform(dataframe, column_name, delete_old):
    transformed_df = dataframe.copy()
    transformed_col = 1 / dataframe[column_name]
    transformed_df.insert(dataframe.columns.get_loc(column_name) + 1, f"{column_name}_inverse", transformed_col)
    if delete_old:
        transformed_df.drop(column_name, axis=1, inplace=True)
    return transformed_df

def rank_transform(dataframe, column_name, delete_old):
    transformed_df = dataframe.copy()
    transformed_col = dataframe[column_name].rank()
    transformed_df.insert(dataframe.columns.get_loc(column_name) + 1, f"{column_name}_rank", transformed_col)
    if delete_old:
        transformed_df.drop(column_name, axis=1, inplace=True)
    return transformed_df

def power_transform(dataframe, column_name, power = 2, delete_old = False):
    transformed_df = dataframe.copy()
    transformed_col = np.power(dataframe[column_name], power)
    transformed_df.insert(dataframe.columns.get_loc(column_name) + 1, f"{column_name}_power_{power}", transformed_col)
    if delete_old == True:
        transformed_df.drop(column_name, axis=1, inplace=True)
    return transformed_df

# Other Functions That Will Contribute In Main Feature





# Now We Will Define Our First Main Feature
class Transform:
    '''
        Only Trainable Features Should Be Passed
        This Feature Will Apply Various Transformation On User Specified Dataset

        Parameter
        ---------
        transform: 
        <str> What Transformation Will Be Used On Datasets, Optional (Default = 'log')
        Possible Values: ['log', 'log1p', '1+log', 'boxcox', 'root', 'exp', 'inverse', 'rank', 'power']

        n_power:
        <float> The Integer Value For That Will Be Used For 'power' Transformation / 'root' Transformation, Optional (Default = 2)

        index_present:
        <bool> User Specifies If The First Column Of Dataset Is An Index Or Any Important Column

        target_split:
        <bool> User Specifies That If User Wants Last Column Splitted As It's Target Column

        delete_old:
        <bool> User Specifies That If This Feature Should Delete Original Columns

        filter_warnings:
        <bool> User Specifies That If This Feature Will Show Warnings Or Not
    '''

    def __init__(self, transform = 'log', n_power = 2, index_present = False, delete_old = False, filter_warnings = False):

        if transform not in ['log', 'log1p', '1+log', 'boxcox', 'root', 'exp', 'inverse', 'rank', 'power']:
            raise ValueError(" transform Should Be One Of Following Values: 'log', 'log1p', '1+log', 'boxcox', 'root', 'exp', 'inverse', 'rank', 'power'")
        if not(isinstance(n_power, (int, float))):
            raise ValueError("n_power Should Be Either Integer Or Float")
        if index_present not in [False, True]:
            raise ValueError('index_present Should Be A <bool>')
        if delete_old not in [False, True]:
            raise ValueError("delete_old Should Be A <bool>")
        if filter_warnings not in [False, True]:
            raise ValueError("filter_warnings Should Be A <bool>")
        if (transform != 'power' and transform != 'root') and (n_power != 2):
            warnings.warn("No Need To Mention n_power If transform Not Equal To 'power' / 'root' ")
        
        self.transform = transform
        self.n_power = n_power
        self.index_present = index_present
        self.delete_old = delete_old
        self.filter_warnings = filter_warnings

    
    def fit_transform(self, Dataset, Col_Names = ['every'], dataframe = True):
        '''
            Only Trainable Features Should Be Passed, This Function Of class Transform, transforms the Given Dataset

            Parameter
            ---------
            Dataset: Pass Dataset

            Col_Names: Mention Col_Names As List, Tuple, Integer, String Or Leave The Decision To The Feature By Passing Default Value = ['every']
            
            Returns
            -------
            NumPy NdArray or Pandas DataFrame
        '''
        if self.filter_warnings:
            warnings.filterwarnings("ignore")

        DataFrame_or = ConvertDataFrame(Dataset, self.index_present)[0]
        DataFrame = DataFrame_or.copy()

        if isinstance(Col_Names, (int, str)):
            if Col_Names not in DataFrame.columns.tolist():
                raise KeyError("Columns Mentioned Not In Dataset")
            Column_Names = [Col_Names,]

        elif isinstance(Col_Names, (list, tuple)) and ((Col_Names != ['every']) and (Col_Names != ('every',))):
            for i in Col_Names:
                if i not in DataFrame.columns.tolist():
                    raise KeyError("Columns Mentioned Not In Dataset")
            Column_Names = Col_Names
        
        elif isinstance(Col_Names, (list, tuple)) and (len(Col_Names) == 1) and (Col_Names[0].lower() == 'every'):
            Column_Names = DataFrame.columns.tolist()
            
        
        # Transformations Will Happen Now
        if self.transform == 'log':
            for column in Column_Names:
                DataFrame = log_transform(DataFrame,column,self.delete_old)
        elif self.transform == 'log1p':
            for column in Column_Names:
                DataFrame = log1p_transform(dataframe=DataFrame, column_name=column, delete_old=self.delete_old)
        elif self.transform == '1+log':
            for column in Column_Names:
                DataFrame = log_plus_1_transform(dataframe=DataFrame, column_name=column, delete_old=self.delete_old)
        elif self.transform == 'boxcox':
            for column in Column_Names:
                DataFrame = boxcox_transform(dataframe=DataFrame, column_name=column, delete_old=self.delete_old)
        elif self.transform == 'root':
            for column in Column_Names:
                DataFrame = root_transform(dataframe=DataFrame, column_name=column, root=self.n_power, delete_old=self.delete_old)
        elif self.transform == 'exp':
            for column in Column_Names:
                DataFrame = exp_transform(dataframe=DataFrame, column_name=column, delete_old=self.delete_old)
        elif self.transform == 'inverse':
            for column in Column_Names:
                DataFrame = inverse_transform(dataframe=DataFrame, column_name=column, delete_old=self.delete_old)
        elif self.transform == 'rank':
            for column in Column_Names:
                DataFrame = rank_transform(dataframe=DataFrame, column_name=column, delete_old=self.delete_old)
        elif self.transform == 'power':
            for column in Column_Names:
                DataFrame = power_transform(dataframe=DataFrame, column_name=column, power=self.n_power, delete_old=self.delete_old)
        else:
            raise ValueError('Wrong transform Passed')
        
        DataFrame = replace_nan_inf(DataFrame)
    
        
        if dataframe:
            return DataFrame
        else:
            NumPy = DataFrameToNumPy(DataFrame)
            return NumPy
        