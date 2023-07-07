import sys
import os

# Get the path of the current file (assuming it is __init__.py)
current_file_path = os.path.abspath(__file__)



# Get the directory path by removing the file name
module_directory = os.path.dirname(current_file_path)

# Add the module directory to sys.path
sys.path.append(module_directory)

from _information import __version__, __desc__, __all__
from _combine_and_split import combine_datasets, split_dataset
from _eda import basic_info, eda, plot_analyze
from _encoding import binary_encoder, category_encoder, one_hot_encoder, date_time_encoder
from _null import TreatNull
from _outliers import FixOutliers, plot_outliers
from _transformation import Transform
