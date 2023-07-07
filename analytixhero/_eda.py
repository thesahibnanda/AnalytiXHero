# Import Libraries
import warnings

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from _basic_functions import ConvertDataFrame, DataFrameToNumPy
from _encoding import category_encoder
from _null import ReplaceEmpty


__all__ = ["basic_info", "eda", "plot_analyze"]




# Functtion Below Will Be Used In Main Feature




# Main Features


# Feature 1
def basic_info(Dataset):
    '''
        This Function Just Prints The Basic Information About Data And Returns The Same

        Parameter
        ---------
        Dataset: User Specified Dataset

        Returns
        -------
        Number of Rows, Column Names, Null Spaces In Each Column, DataType Of Each Column
    '''

    DataFrame = ConvertDataFrame(Dataset, False)[0]

    row = DataFrame.shape[0]
    col = DataFrame.columns.tolist()
    dtype = DataFrame.dtypes
    DataFrame = ReplaceEmpty(DataFrame=DataFrame)
    Null = DataFrame.isnull().sum()

    print('Number of Rows:',row)
    print()
    print()
    print('Column List:', col)
    print()
    print()
    print("Data Types:")
    print(dtype)
    print()
    print()
    print('Null Spaces Of Every Data Type:')
    print(Null)

    return row, col, dtype, Null

# Feature 2
def eda(Dataset, target_present = False, show_message = True):
    '''
        Prints All Types Of Mathematical Data Analysis

        Parameter
        ---------
        Dataset: User Specifies Dataset

        Returns
        -------
        None, It Just Prints Everything
    '''

    warnings.filterwarnings('ignore')
    df_or, columns_list = ConvertDataFrame(Dataset, False)
    

    df = df_or

    df = category_encoder(df)
    df = ConvertDataFrame(df, False)[0]
    df.columns = columns_list
    
    if show_message:
        print("It's Advised To Do Proper Encoding Before EDA")
        print("Note: Skewness and Kurtosis Returns NaN or Null Value If Data Is Nearly Identical")
    

    if target_present:
        df = df.iloc[:, :-1]

    for col in df.columns:
        print(f"Column Name: {col}")
        print(f"Mean: {df[col].mean()}")
        print(f"Median: {df[col].median()}")
        print(f"Mode: {df[col].mode()[0]}")
        print(f"Skewness: {skew(df[col], nan_policy='omit')}")
        print(f"Kurtosis: {kurtosis(df[col], nan_policy='omit')}")
        print(f"Minimum: {df[col].min()}")
        print(f"Maximum: {df[col].max()}")
        print(f"Range: {df[col].max() - df[col].min()}")
        print(f"Standard deviation: {df[col].std()}")
        print()
        print()
    if target_present:
        print('Target Is Mathematically Invalid, Even If It Is Not In Object Dtype')

# Feature 3
def plot_analyze(data,  Col_Names=['every'], basis='standard', index_present=False):
    '''
        Plot Graphs On Different Basis To Analyze Data

        Parameter
        ---------
        data: User Specifies Dataset

        col_names: User Specifies col_names as list, tuple, int and str, Optional (Default: ['every'] -> All Columns)

        basis: On What Basis Graphs Will Be Plotted

        index_present: User Specifies if there is an index column in Dataset <bool>, Optional (Default: False)

        Returns
        -------
        None, Just Plots Graphs
    '''
    if basis not in ['corr', 'standard', 'skew', 'kurtosis', 'var', 'linear']:
        raise ValueError("Invalid basis, basis Should Be In ['corr', 'standard', 'skew', 'kurtosis', 'var', 'linear']")
    
    col_names = Col_Names

    dataframe_or = ConvertDataFrame(data, index_present=index_present)[0]
    dataframe = dataframe_or.copy()

    if isinstance(col_names, (int, str)):
        if col_names not in dataframe.columns.tolist():
            raise KeyError("Columns mentioned not in dataset")
        column_names = [col_names]
    elif isinstance(col_names, (list, tuple)) and ((col_names != ['every']) and (col_names != ('every',))):
        for i in col_names:
            if i not in dataframe.columns.tolist():
                raise KeyError("Columns mentioned not in dataset")
        column_names = col_names
    elif isinstance(col_names, (list, tuple)) and (len(col_names) == 1) and (col_names[0].lower() == 'every'):
        column_names = dataframe.columns.tolist()
    else:
        raise ValueError("Invalid column names provided")

    if basis == 'corr':
        corr_matrix = dataframe[column_names].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title(f'Correlation Heatmap For All Given Columns: {column_names}')
        plt.show()
    elif basis == 'standard':
        fig, axs = plt.subplots(len(column_names), figsize=(10, 5 * len(column_names)))
        plt.subplots_adjust(hspace=0.5)

        for i, col in enumerate(column_names):
            ax = axs[i] if len(column_names) > 1 else axs
            sns.histplot(dataframe[col], kde=True, ax=ax)
            ax.set_title(f'Graph For Column: {col}')

        plt.show()
    else:
        fig, axs = plt.subplots(len(column_names), figsize=(10, 5 * len(column_names)))
        plt.subplots_adjust(hspace=0.5)

        for i, col in enumerate(column_names):
            ax = axs[i] if len(column_names) > 1 else axs
            if basis == 'skew':
                sns.histplot(dataframe[col], kde=True, ax=ax)
                ax.set_title(f'Skewness For Column: {col}')
            elif basis == 'kurtosis':
                sns.histplot(dataframe[col], kde=True, ax=ax)
                ax.set_title(f'Kurtosis For Column: {col}')
            elif basis == 'var':
                ax.plot(dataframe[col])
                ax.set_title(f'Variance For Column: {col}')
            elif basis == 'linear':
                sns.scatterplot(x=dataframe.index, y=dataframe[col], ax=ax)
                ax.set_title(f'Linearity For Column: {col}')
            else:
                raise ValueError("Invalid basis provided")

        plt.show()

