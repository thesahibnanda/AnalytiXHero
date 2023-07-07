# Import Required Libraries
import warnings

import numpy as np
import pandas as pd

from _basic_functions import ConvertDataFrame, DataFrameToNumPy

__all__ = ["combine_datasets", "split_dataset"]

# This Function Will Be A Main Feature, Stacking Columns If User Uploads Data After Spliting To A Target

def combine_datasets(Train_Dataset, Target_Dataset):

    """
        Combines the given Train_Dataset and Target_Dataset into a single DataFrame.

        Args:
            Train_Dataset: The training dataset.
            Target_Dataset: The test dataset.

        Returns:
            tuple: A tuple containing the combined DataFrame and the column names.

        Raises:
            ValueError: If the Train_Dataset or Target_Dataset is not in the correct format.
                - If Target_Dataset has more than one column or the number of data points in
                - Train_Dataset and Target_Dataset is not equal.
    """
    Train_Dataset = ConvertDataFrame(Train_Dataset, False)[0]
    Target_Dataset = ConvertDataFrame(Target_Dataset, False)[0]
    CombinedDataFrame = pd.concat([Train_Dataset, Target_Dataset], axis=1)
    if (Target_Dataset.shape[1] != 1) or (Train_Dataset.shape[0] != Target_Dataset.shape[0]):
        raise ValueError("Train/Test Dataset Passed Isn't Correct. Either Test Data Has More Than 1 Column Or Datapoints In Both Test And Train Dataset Aren't Equal")
    CombinedDataFrame  = pd.DataFrame(CombinedDataFrame)
    CombinedNumPy = DataFrameToNumPy(CombinedDataFrame)
    
    return CombinedNumPy

def split_dataset(Full_Dataset):
    '''
        Splits The Given Full_Dataset In To X and y

        Args:
            Full_Dataset: Dataset Given By User
        
        Returns:
            X Dataset and y Dataset
    '''

    Full_Dataset = ConvertDataFrame(Full_Dataset, False)[0]

    X = Full_Dataset.iloc[:, :-1]  # All columns except the last column   
    y = Full_Dataset.iloc[:, -1]   # Last column

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    X = DataFrameToNumPy(X)
    y = DataFrameToNumPy(y)

    return X, y
