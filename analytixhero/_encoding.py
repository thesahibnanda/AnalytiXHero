# Import Required Libraries
import warnings

import numpy as np
import pandas as pd
from dateutil import parser

from _basic_functions import ConvertDataFrame, DataFrameToNumPy

__all__ = ["binary_encoder", "category_encoder", "one_hot_encoder","date_time_encoder"]


def binary_encoder(Dataset, Col_Names=['auto']):
    '''
        Binary_Encoder Encodes Each Column With 2 Unique Values To 0 And 1

        Parameter
        ---------
            Dataset: User Enter Dataset

            Col_Names: User Specifies Column Names To Undergo Encoding, Default Value is ['auto'] That Automatically Selects Columns
        Returns
        -------
            Encoded Numpy NdArray
    '''


    DataFrame_or = ConvertDataFrame(Dataset, False)[0]
    DataFrame = DataFrame_or.copy()

    if (isinstance(Col_Names, int) or isinstance(Col_Names, str)) and (Col_Names in DataFrame.columns.tolist()):
        Column_Names = [Col_Names,]
    elif (isinstance(Col_Names, list) or isinstance(Col_Names, tuple)) and ((Col_Names != ['auto']) and (Col_Names != ('auto',))):
        for i in Col_Names:
            if i not in DataFrame.columns.tolist():
                raise ValueError("Columns Not In DF")
        Column_Names = Col_Names
    elif (isinstance(Col_Names, list) or isinstance(Col_Names, tuple)) and (len(Col_Names) == 1) and Col_Names[0].lower() == 'auto':
        Column_Names = DataFrame.columns.tolist()
    else:
      raise ValueError("Columns Defined Aren't In Required Way")

    for column in Column_Names:
        if DataFrame[column].nunique() == 2:
            Value_List = DataFrame[column].unique()
            Value_Dict = {}

            for i in range(len(Value_List)):
                Value_Dict[Value_List[i]] = i


            DataFrame[column] = DataFrame[column].map(Value_Dict)
    
    Numpy = DataFrameToNumPy(DataFrame)
    return Numpy



def category_encoder(Dataset, Col_Names=['auto']):
    '''
        Category_Encoder Encodes Each Column With Object Dtype To Integer Encoding

        Parameter
        ---------
            Dataset: User Enter Dataset

            Col_Names: User Specifies Column Names To Undergo Encoding, Default Value is ['auto'] That Automatically Selects Columns
        Returns
        -------
            Encoded Numpy NdArray
    '''

    DataFrame_or = ConvertDataFrame(Dataset, False)[0]
    DataFrame = DataFrame_or.copy()

    if (isinstance(Col_Names, int) or isinstance(Col_Names, str)):
        if Col_Names not in DataFrame.columns.tolist():
            raise ValueError("Columns Not In Dataset")
        Column_Names = [Col_Names,]
    elif (isinstance(Col_Names, (list, tuple))) and ((Col_Names != ['auto']) and (Col_Names != ('auto',))):
        for i in Col_Names:
            if i not in DataFrame.columns.tolist():
                raise ValueError("Columns Not In Dataset")
        Column_Names = Col_Names
    elif (isinstance(Col_Names, (list, tuple))) and (len(Col_Names) == 1) and Col_Names[0].lower() == 'auto':
        Column_Names = DataFrame.columns.tolist()
    else:
        raise ValueError("Columns Defined Aren't In Required Way")

    for column_name in Column_Names:
        column_values = DataFrame[column_name].unique()

        # Check if the column can be converted to a datetime column
        try:
            pd.to_datetime(DataFrame[column_name])
            is_datetime = True
        except:
            is_datetime = False

        # Check if the column contains categorical values and is not a datetime column
        if ((DataFrame[column_name].dtype == 'object' or np.issubdtype(DataFrame[column_name].dtype, np.bool_))) and ((not is_datetime) and (not np.issubdtype(DataFrame[column_name].dtype, np.datetime64))):
            Value_Dict = {}

            for i in range(len(column_values)):
                Value_Dict[column_values[i]] = i

            DataFrame[column_name] = DataFrame[column_name].map(Value_Dict)

    Numpy = DataFrameToNumPy(DataFrame)
    return Numpy


def one_hot_encoder(Dataset, Col_Names=['auto'], delete_old=False, label_data_present = False, dataframe = True):
    """
        Perform one-hot encoding on categorical columns of a dataset.

        Parameters
        ----------
            - Dataset (DataFrame or ndarray): The input dataset to encode.
            - Col_Names (int, str, list, tuple): Specifies the columns to encode. Default is 'auto', which encodes all columns.
            - delete_old (bool): Determines whether to delete the original columns after encoding. Default is False.
            - label_data_present (bool): Specifies whether the dataset includes label data. Default is False.
            - dataframe (bool): Determines the output format. If True, returns a DataFrame. If False, returns a NumPy ndarray. Default is False.

        Returns
        -------
            - DataFrame or ndarray: The encoded dataset based on the specified parameters.
    """

    DataFrame_or = ConvertDataFrame(Dataset, False)[0]
    DataFrame = DataFrame_or.copy()

    if (isinstance(Col_Names, int) or isinstance(Col_Names, str)):
        if Col_Names in DataFrame.columns.tolist():
            raise ValueError("Columns Not In Datset")
        Column_Names = [Col_Names,]
    elif (isinstance(Col_Names, (list, tuple))) and ((Col_Names != ['auto']) and (Col_Names != ('auto',))):
        for i in Col_Names:
            if i not in DataFrame.columns.tolist():
                raise ValueError("Columns Not In Dataset")
        Column_Names = Col_Names
    elif (isinstance(Col_Names, (list, tuple))) and (len(Col_Names) == 1) and (Col_Names[0].lower() == 'auto') and (label_data_present == False):
        Column_Names = DataFrame.columns.tolist()
    elif (isinstance(Col_Names, (list, tuple))) and (len(Col_Names) == 1) and (Col_Names[0].lower() == 'auto') and (label_data_present == True):
        Column_Names = DataFrame.columns.tolist()
        Column_Names = Column_Names[0:-1]
    else:
        raise ValueError("Columns Defined Aren't In Required Way")

    for column in Column_Names:
        c = DataFrame.columns.tolist().index(column) + 1  # Get the index of the original column
        
        # Pandas To DateTime Conversion
        try:
            pd.to_datetime(DataFrame[column])
            is_datetime = True
        except:
            is_datetime = False


        # Check if column is categorical or not
        if (DataFrame[column].dtype == 'object' or np.issubdtype(DataFrame[column].dtype, np.bool)) and ((not is_datetime) and (not np.issubdtype(DataFrame[column].dtype, np.datetime64))):
            Values_List = DataFrame[column].unique()
            for Value in Values_List:
                v = 0  # Initialize the counter for the new encoded columns
                Values_Dict = {}
                for i in range(len(Values_List)):
                    if Value == Values_List[i]:
                        Values_Dict[Value] = 1
                    else:
                        Values_Dict[Values_List[i]] = 0

                index = c+v
                DataFrame.insert(index, str(column) + "_" + str(Value), DataFrame[column].map(Values_Dict))

            if delete_old:
                DataFrame.drop(column, axis=1, inplace=True)

    if dataframe: # Output As DataFrame
        return DataFrame
    else: # Output As NumPy NDArray

        NumPy = DataFrameToNumPy(DataFrame)
        return NumPy

def date_time_encoder(Dataset, Col_Names=['auto'], delete_old=False, dataframe=True):
    """
    Encode date or datetime columns in a dataset by creating new columns representing various components of the date and time.
    
    Parameters
    ----------
    - Dataset (DataFrame, array-like): The input dataset to encode.
    - Col_Names (int, str, list, tuple): Specifies the columns to encode. Default is 'auto', which encodes all date or datetime columns.
    - delete_old (bool): Specifies whether to delete the original date or datetime columns after encoding. Default is False.
    - dataframe (bool): Specifies whether to return the encoded data as a DataFrame. If False, the encoded data is returned as a NumPy array. Default is False.
    
    Returns
    -------
    - If dataframe=True: Returns the encoded dataset as a DataFrame.
    - If dataframe=False: Returns the encoded dataset as a NumPy array.
    """

    DataFrame_or = ConvertDataFrame(Dataset, False)[0]

    DataFrame = DataFrame_or.copy()

    if isinstance(Col_Names, int) or isinstance(Col_Names, str):
        Col_Names = [Col_Names, ]
    elif isinstance(Col_Names, (list, tuple)):
        if Col_Names != ['auto'] and Col_Names != ('auto',):
            for i in Col_Names:
                if i not in DataFrame.columns.tolist():
                    raise ValueError("Columns Not In DataFrame")
        else:
            Col_Names = DataFrame.columns.tolist()
    else:
        raise ValueError("Columns Defined Aren't In Required Way")

    to_be_column = []
    for column in Col_Names:
        if column not in DataFrame.columns.tolist():
            raise ValueError("Column Not Found in DataFrame")

        # Check if column is in date or datetime format or can be converted to datetime
        if DataFrame[column].dtype == 'object':
            try:
                DataFrame[column] = DataFrame[column].apply(parser.parse)
                to_be_column.append(column)
            except ValueError:
                pass
        elif np.issubdtype(DataFrame[column].dtype, np.datetime64):
            to_be_column.append(column)
            pass
        
        
        # Check if column is in date-time format
        if DataFrame[column].dtype == np.dtype('datetime64[ns]'):
            
            new_date_time_columns = [
                column + "_year",
                column + "_month",
                column + "_day",
                column + "_hour",
                column + "_minute",
                column + "_second"]
            c = DataFrame.columns.get_loc(column)
            DataFrame.insert(c+1, new_date_time_columns[0], DataFrame[column].dt.year)
            DataFrame.insert(c+2, new_date_time_columns[1], DataFrame[column].dt.month)
            DataFrame.insert(c+3, new_date_time_columns[2], DataFrame[column].dt.day)
            DataFrame.insert(c+4, new_date_time_columns[3], DataFrame[column].dt.hour)
            DataFrame.insert(c+5, new_date_time_columns[4], DataFrame[column].dt.minute)
            DataFrame.insert(c+6, new_date_time_columns[5], DataFrame[column].dt.second)

            for i in new_date_time_columns:
                if DataFrame[i].tolist() == ([0]*len(DataFrame[i])):
                    DataFrame.drop(i, axis=1, inplace=True)

    if delete_old:
        DataFrame.drop(to_be_column, axis=1, inplace=True)

    if dataframe == True:
        return DataFrame
    elif dataframe == False:
        Numpy = DataFrameToNumPy(DataFrame)
        return Numpy