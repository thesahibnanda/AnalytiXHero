# Import Required Libraries
import warnings

import numpy as np
import pandas as pd


# Function 1, Converting DataFrame To NumPy NDArray
def DataFrameToNumPy(df):
    if isinstance(df, pd.DataFrame):
        if df.empty:
            return np.array([])  # Return an empty ndarray if DataFrame is empty
        else:
            return df.to_numpy()  # Convert DataFrame to ndarray
    else:
        raise ValueError("Input must be a DataFrame.")
                

# Function 2, Convert Any Data To Pandas DataFrame
def ConvertDataFrame(data, index_present=False):
    """
    Converts various data structures to a Pandas DataFrame.

    Args:
        data: The input data structure to be converted to a DataFrame. Supported types are:
            - list: Converted to DataFrame with column names assigned as '1', '2', '3', ...
            - tuple: Converted to DataFrame with column names assigned as '1', '2', '3', ...
            - np.ndarray: Converted to DataFrame with column names assigned as '1', '2', '3', ...
            - dict: Converted to DataFrame with keys as column names and values as column data
            - pd.DataFrame: Returned as-is without any modification

        index_present (bool, optional): Indicates whether the resulting DataFrame should have an index column.
            If set to True, the first column of the DataFrame will be removed. Default is False.

    Returns:
        df (pd.DataFrame): The converted Pandas DataFrame.
        column_names (list): The list of column names assigned to the DataFrame.

    Raises:
        ValueError: If the data structure is not supported or unrecognized.
    """
    if not isinstance(data, (list, tuple, np.ndarray, dict, pd.DataFrame)):
        raise ValueError("Unsupported data structure. Only lists, tuples, dictionaries, and NumPy arrays are supported.")

    if isinstance(data, pd.DataFrame):
        if index_present:
            data = data.iloc[:, 1:]  # Remove the first column if index_present is True
        return data, data.columns.tolist()

    df = pd.DataFrame(data)

    if not df.columns.tolist() or None in df.columns.tolist():  # Empty or missing column names
        column_names = [str(i) for i in range(0, df.shape[1])]
        df.columns = column_names

    df = df.iloc[:, 1:] if index_present else df

    return df, df.columns.tolist()