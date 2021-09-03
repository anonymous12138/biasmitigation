from aif360.algorithms.preprocessing.reweighing import Reweighing
import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.tree import _tree
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath('..'))

from imblearn.over_sampling import SMOTE
from Measure import *
# from Generate_Samples import generate_samples

import aif360
from aif360.datasets import BinaryLabelDataset, StructuredDataset, StandardDataset
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult,get_distortion_german,get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult,load_preproc_data_german,load_preproc_data_compas
from IPython.display import Markdown, display
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric