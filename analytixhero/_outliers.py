#Importing Required Libraries
import warnings

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from _basic_functions import ConvertDataFrame, DataFrameToNumPy

__all__ = ["FixOutliers", "plot_outliers"]


# Now, Those Functions Will Be Defined That Will Be Used In Our Main Feature Called 'FixOutliers'




# Function 1.1, Removal Of Outliers Via Z Score's Approach
def Outliers_Removal_Z_Score(df, col_name, threshold):
    """
        Parameters
        ----------
        df: 'pandas DataFrame'
        DataFrame From Which Outliers Need To Be Detected Is Passed

        col_name: 'str'
        Name Of The Column That Needs To Undergo Outlier Detection And Removal Is Passed

        threshold: 'int'
        Threshold Value For Outlier Computation Is Passed

        Returns
        -------
        df: 'pandas DataFrame'
        DataFrame After Removal Of Outliers Is Returned
    """

    #Calculate Z - Score
    z_scores = np.abs((df[col_name] - df[col_name].mean()) / df[col_name].std())

    #Identify Outliers
    outliers = z_scores > threshold

    #Removal
    df = df[~outliers]

    return df

# Function 1.2, Removal Of Outliers Via Inter Quartile Range Approach
def Outliers_Removal_IQR(df, col_name):
    """
        Parameters
        ----------
        df: 'pandas DataFrame'
        DataFrame From Which Outliers Need To Be Detected Is Passed

        col_name: 'str'
        Name Of The Column That Needs To Undergo Outlier Detection And Removal Is Passed

        Returns
        -------
        df: 'pandas DataFrame'
        DataFrame After Removal Of Outliers Is Returned
    """

    #Calculate First(25 Percentile) And Third(75 Percentile) Quartile
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)

    #Calculating InterQuartile Range
    IQR = Q3 - Q1

    #Defining Upper Bound And Lower Bound
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    #Removing Outliers
    df = df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]

    return df

# Function 2.1, Imputing Outliers Via Z Score Approach
def Impute_Z_Score(df, col_name, method, consider_outliers, threshold):
    """
        Parameters
        ----------
        df: 'pandas DataFrame'
        DataFrame From Which Outliers Need To Be Detected Is Passed

        col_name: 'str'
        Name Of The Column That Needs To Undergo Outlier Detection And Removal Is Passed

        method: 'str'
        What Value Should Be Used For Imputation - Mean, Median, Mode Or Std. Deviation

        consider_outliers: 'bool'
        Should Outliers Be Considered While Calculating Imputation Values

        threshold: 'int'
        Threshold Value For Outlier Computation Is Passed

        Returns
        -------
        df: 'pandas DataFrame'
        DataFrame After Imputation
    """
    
    #Convert Method In Lower Case
    method = method.lower()
        
    #Finding Outliers In The DataFrame
    
    #Calculate Z - Score
    z_scores = np.abs((df[col_name] - df[col_name].mean()) / df[col_name].std())
    
    #Identify Outliers
    outliers = z_scores > threshold
    
    #Considering Outliers
    if consider_outliers:
        if method == 'mean':
            impute = df[col_name].mean()
            df.loc[outliers, col_name] = impute
        elif method == 'median':
            impute = df[col_name].median()
            df.loc[outliers, col_name] = impute
        elif method == 'mode':
            impute = stats.mode(df[col_name] , keepdims = True)[0][0]
            df.loc[outliers, col_name] = impute
        elif method == 'std':
            impute = df[col_name].std()
            df.loc[outliers, col_name] = impute
        else:
            raise ValueError('Invalid Imputation, Valid Options: Mean, Median, Mode Or Std')
    else:
        if method == 'mean':
            impute = df.loc[~outliers, col_name].mean()
            df.loc[outliers, col_name] = impute
        elif method == 'median':
            impute = df.loc[~outliers, col_name].median()
            df.loc[outliers, col_name] = impute
        elif method == 'mode':
            impute = stats.mode(df.loc[~outliers, col_name], keepdims = True)[0][0]
            df.loc[outliers, col_name] = impute
        elif method == 'std':
            impute = df.loc[~outliers, col_name].std()
            df.loc[outliers, col_name] = impute
        else:
            raise ValueError('Invalid Imputation, Valid Options: Mean, Median, Mode Or Std')
            
    return df

# Function 2.2, Imputing Outliers Via InterQuartile Range
def Impute_IQR(df, col_name, method, consider_outliers):
    """
        Parameters
        ----------
        df: 'pandas DataFrame'
        DataFrame From Which Outliers Need To Be Detected Is Passed

        col_name: 'str'
        Name Of The Column That Needs To Undergo Outlier Detection And Removal Is Passed

        method: 'str'
        What Value Should Be Used For Imputation - Mean, Median, Mode Or Std. Deviation

        consider_outliers: 'bool'
        Should Outliers Be Considered While Calculating Imputation Values

        Returns
        -------
        df: 'pandas DataFrame'
        DataFrame After Imputation
    """
    
    #Convert Method In Lower Case
    method = method.lower()
        
    #Finding Outliers In The DataFrame
    
    #Calculate IQR
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3 - q1
    
    #Identify Outliers
    outliers = (df[col_name] < (q1 - 1.5*iqr)) | (df[col_name] > (q3 + 1.5*iqr))
    
    #Considering Outliers
    if consider_outliers:
        if method == 'mean':
            impute = df[col_name].mean()
            df.loc[outliers, col_name] = impute
        elif method == 'median':
            impute = df[col_name].median()
            df.loc[outliers, col_name] = impute
        elif method == 'mode':
            impute = stats.mode(df[col_name], keepdims = True)[0][0]
            df.loc[outliers, col_name] = impute
        elif method == 'std':
            impute = df[col_name].std()
            df.loc[outliers, col_name] = impute
        else:
            raise ValueError('Invalid Imputation, Valid Options: Mean, Median, Mode Or Std')
    else:
        if method == 'mean':
            impute = df.loc[~outliers, col_name].mean()
            df.loc[outliers, col_name] = impute
        elif method == 'median':
            impute = df.loc[~outliers, col_name].median()
            df.loc[outliers, col_name] = impute
        elif method == 'mode':
            impute = stats.mode(df.loc[~outliers, col_name], keepdims = True)[0][0]
            df.loc[outliers, col_name] = impute
        elif method == 'std':
            impute = df.loc[~outliers, col_name].std()
            df.loc[outliers, col_name] = impute
        else:
            raise ValueError('Invalid Imputation, Valid Options: Mean, Median, Mode Or Std')
            
    return df


# Now We'll Define Our First Main Feature
class FixOutliers:
    '''
        Used To Remove Or Impute Outliers According To The Approaches (To Detect Outliers) Defined By The User
        Remove Means Deleting The Datapoints Or Rows That Have Outliers
        Impute Means Replacing Values Of Outliers With Some Other Value Like Mean, Median, Mode.
    '''

    '''
        Parameters
        ----------
            approach: 'str', Optional (Default = 'interquartile_range')
            User Specifies The Approach He/She Wants To Be Executed (Possible Values = ['interquartile_range', 'z_score', 'all'])

            treatment: 'str', Optional (Default = 'impute')
            User Specifies What Treatment Will Outliers Undergo (Possible Values = ['impute','remove'])

            imputation: 'str', Optional (Default = 'mean')
            User Specifies Which Value Will Be Used For Imputation (Possible Values = ['mean', 'median', 'mode', 'std'])
            Only Useful If treatment = 'impute'

            consider_outliers: 'bool', Optional (Default = False)
            User Specifies Should Machine Consider Outliers While Computing Imputation Value
            Only Useful If treatment = 'impute'

            threshold: 'int', Optional (Default = 3)
            User Specifies Threshold For Z-Score Outlier Computation
            Only Useful If self.approach = 'z_score'

            index_present: 'bool', Optional (Default = False)
            User Specifies Whether The First Column Of The Dataset Is Just An Index Column Or An Important Column

            target_split: 'bool', Optional (Default = False)
            User Specifies Whether The Last Column Of The Dataset Is A Target Column And Whether User Wants It To Be Returned As A Different Variable
        
        Return
        ------
            numpy.ndarray After Outlier Treatment
    '''

    def __init__(
            self,
            approach = 'interquartile_range',
            treatment = 'impute',
            imputation = 'mean',
            consider_outliers = False,
            threshold = 3,
            index_present = False,
            target_split = False
            ):
        
            if approach not in ['z_score', 'interquartile_range', 'all']:
                raise ValueError('Invalid Approach Parameter')

            if treatment not in ['remove', 'impute']:
                raise ValueError("Unknown Value Passed As treatment")
            
            if imputation not in ['mean', 'median', 'mode', 'std']:
                raise ValueError("Unknown Value Passed As imputation")
        
            if consider_outliers not in [True, False]:
                raise ValueError("consider_outliers Needs To <'bool'>")
            
            if index_present not in [True, False]:
                raise ValueError("index_present Needs To <'bool'>")
            
            if target_split not in [True, False]:
                raise ValueError("index_present Needs To <'bool'>")

            if treatment == 'remove' and (imputation != 'mean' or consider_outliers != False):
                warnings.warn(message="No Need Of imputation or consider_outliers when treatment = 'remove'")

            if approach == 'interquartile_range' and threshold != 3:
                warnings.warn("InterQuartile Range Does Not Require Threshold")
            
            self.approach = approach
            self.treatment = treatment
            self.imputation = imputation
            self.consider_outliers = consider_outliers
            self.threshold = threshold
            self.index_present = index_present
            self.target_split = target_split
    

    def fit_transform(self, Dataset, Col_Names = ['every']):

        '''
            Parameters
            ----------
                Dataset: This Dataset Will Undergo Outlier Treatment
                Any Supported Data Structure (List, Tuple, Dictionary, Numpy.Ndarray, Pandas.DataFrame)

                Col_Name: Only These Columns Will Undergo Outlier Treatment (Default: ['every'])
                User Can Either Specify Multiple Columns As A 1-D Tuple/List
                Or A String Or An Integer 
                If He/She Wishes To Pass Only One Column Name
                If User Passes ['every'], Then Every Column Will Undergo Outlier Treatment
                Col_Names Other Than <int> and 'every Only Applicable If Dataset Is <dict> or <pandas.DataFrame>
        '''

        # Converting Dataset In Pandas DataFrame
        Dataset_or, Column_Names = ConvertDataFrame(Dataset, self.index_present)

        Dataset = Dataset_or.copy()
    
        if isinstance(Col_Names, int):
            Column_Names = (Col_Names,)
        elif isinstance(Col_Names, str):
            if isinstance(Dataset, pd.DataFrame):
                Column_Names = (Col_Names,)
            else:
                warnings.warn("No Use Of Providing <str> Column Names")
        elif (isinstance(Col_Names, list) or isinstance(Col_Names, tuple)) and Col_Names != ['every']:
            if (set(Col_Names) <= set(Column_Names)):
                Column_Names = Col_Names
        elif isinstance(Col_Names, list) and len(Col_Names) == 1 and Col_Names[0].lower() == 'every':
            Column_Names = Column_Names
        else:
            raise ValueError("Column Name(s) Not Specified As Per Required")


        if self.treatment == 'remove':
            if self.approach == 'interquartile_range':
                for i in Column_Names:
                    Dataset = Outliers_Removal_IQR(Dataset, i)
            elif self.approach == 'z_score':
                for i in Column_Names:
                    Dataset = Outliers_Removal_Z_Score(Dataset, i, self.threshold)
            else:
                for i in Column_Names:
                    Dataset = Outliers_Removal_IQR(Dataset, i)
                for i in Column_Names:
                    Dataset = Outliers_Removal_Z_Score(Dataset, i, self.threshold)
        elif self.treatment == 'impute':
            if self.approach == 'interquartile_range':
                for i in Column_Names:
                    Dataset = Impute_IQR(Dataset,i,method=self.imputation, consider_outliers=self.consider_outliers)
            elif self.approach == 'z_score':
                for i in Column_Names:
                    Dataset = Impute_Z_Score(Dataset, i, method=self.imputation, consider_outliers=self.consider_outliers, threshold=self.threshold)
            else:
                for i in Column_Names:
                    Dataset = Impute_IQR(Dataset,i,method=self.imputation, consider_outliers=self.consider_outliers)
                for i in Column_Names:
                    Dataset = Impute_Z_Score(Dataset, i, method=self.imputation, consider_outliers=self.consider_outliers, threshold=self.threshold)


        if self.target_split:
            X = pd.DataFrame(Dataset.iloc[:, :-1])
            y = pd.DataFrame(Dataset.iloc[:, -1])
            
            X = DataFrameToNumPy(X)
            y = DataFrameToNumPy(y)

            return X, y
        else:
            Dataset = pd.DataFrame(Dataset)
            Dataset = DataFrameToNumPy(Dataset)

            return Dataset
    

# Approaches Used In Writing This Feature

"""
    Approaches To Detect Outliers: 

    Approach 1 - On The Basis Of Z Score
    Z = (Datapoint - Mean) / Standard Deviation
    Find The Data Points Whose Z-Scores Are Higher Than The Specified Threshold Value (Default Threshold Value Is 3)

    Approach 2 - On The Basis Of InterQuartile Range
    Q1 Is The Point That Accounts 25 Percentile Of All Data
    Q2 Is The Point That Accounts 50 Percentile Of All Data
    Q3 Is The Point That Accounts 75 Percentile Of All Data
    Interquartile Range = Q3 - Q1
    Minimum of Quartile = Q1 - (1.5 X Interquartile Range)
    Maximum of Quartile = Q3 + (1.5 X Interquartile Range)
    Data Points Not In Range Of Minimum of Quartile And Maximum of Quartile Are Outliers 

    Approach 3 - On The Basis Of Both Approaches Mentioned Above
    First Outliers Will Detected And Taken Care Of Using 1st Approach Then 2nd Approach
"""


# Now We'll Define Our Second Main Feature
def plot_outliers(dataset, Col_Names=['every'], fontsize = 12, z_threshold=3, db_eps=0.5, db_min_samples=2, if_contamination=0.01, color=['red', 'blue', 'green', 'cyan'], index_present = False):

    """
        Plot outliers in a scatter plot for the specified columns in a Dataset.

        Parameters
        ----------
            - dataset (DataFrame): The input Dataset containing the data.
            - Col_Names (int, str, list, tuple): Specifies the columns to plot. Default is ['every'], which plots all columns.
            - fontsize (int): Specifies fontsize for marking outliers
            - z_threshold (float): The threshold for Z-score method. Default is 3.
            - db_eps (float): The maximum distance between two samples for DBSCAN clustering. Default is 0.5.
            - db_min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point in DBSCAN. Default is 2.
            - if_contamination (float): The estimated proportion of outliers in the data for the Isolation Forest method. Default is 0.01.
            - color (list, tuple, str): The colors to use for plotting the outliers. Default is ['red', 'blue', 'green', 'cyan'].
            - index_present (bool): If index present in the Dataset as first column. Default is False.

        Returns
        -------
            None But Plots Graphs
    """

    # Suppress Warnings By Sklearn
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # Dataset To DataFrame
    dataframe = ConvertDataFrame(dataset, index_present=index_present)[0]


    # Column Selection
    if (isinstance(Col_Names, int) or isinstance(Col_Names, str)):
        if Col_Names not in dataframe.columns.tolist():
            raise ValueError("Columns Not In DataFrame")
        column_names = [Col_Names]
    elif isinstance(Col_Names, (list, tuple)) and Col_Names != ['every'] and Col_Names != ('every',):
        for col in Col_Names:
            if col not in dataframe.columns.tolist():
                raise ValueError("Columns Not In DataFrame")
        column_names = Col_Names
    elif isinstance(Col_Names, (list, tuple)) and len(Col_Names) == 1 and Col_Names[0].lower() == 'every':
        column_names = dataframe.columns.tolist()
    else:
        raise ValueError("Columns Defined Aren't In the Required Way")

    # Color Initialization
    if isinstance(color, (list, tuple)) and len(color) == 4:
        color = color
    elif isinstance(color, str):
        color = [color, color, color, color]

    # Create a figure and axis object
    if len(column_names) == 1:
        fig, axes = plt.subplots(nrows=1, figsize=(10, 5))
        axes = [axes]  # Convert axes to a list
    else:
        fig, axes = plt.subplots(nrows=len(column_names), figsize=(10, 5 * len(column_names)))

    # Iterate over each column in the DataFrame
    for i, column in enumerate(column_names):
        # Create a scatter plot for the column
        ax = axes[i]
        sns.scatterplot(x=dataframe.index, y=column, data=dataframe, ax=ax)

        # Approach 1: Z-score method
        z_scores = (dataframe[column] - dataframe[column].mean()) / dataframe[column].std()
        z_outliers = dataframe[abs(z_scores) > z_threshold]
        for index, row in z_outliers.iterrows():
            ax.text(index, row[column], 'o', fontsize=fontsize, color=color[0], ha='center', va='center')

        # Approach 2: Interquartile Range (IQR) method
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = dataframe[(dataframe[column] < q1 - 1.5 * iqr) | (dataframe[column] > q3 + 1.5 * iqr)]
        for index, row in iqr_outliers.iterrows():
            ax.text(index, row[column], 'o', fontsize=fontsize, color=color[1], ha='center', va='center')

        # Approach 3: Clustering algorithm (DBSCAN) method
        X = dataframe[[column]]
        clustering = DBSCAN(eps=db_eps, min_samples=db_min_samples).fit(X)
        cluster_labels = clustering.labels_
        cluster_outliers = dataframe[cluster_labels == -1]
        for index, row in cluster_outliers.iterrows():
            ax.text(index, row[column], 'o', fontsize=fontsize, color=color[2], ha='center', va='center')

        # Approach 4: Isolation Forest method
        isolation_forest = IsolationForest(contamination=if_contamination, random_state=0)
        isolation_forest.fit(X)
        if_outliers = dataframe[isolation_forest.predict(X) == -1]
        for index, row in if_outliers.iterrows():
            ax.text(index, row[column], 'o', fontsize=fontsize, color=color[3], ha='center', va='center')

        # Set the x-axis label
        ax.set_xlabel('Index')

        # Set the y-axis label
        ax.set_ylabel(column)

        # Set the title for the subplot
        ax.set_title(f'Scatter plot with outliers for {column}')

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()
