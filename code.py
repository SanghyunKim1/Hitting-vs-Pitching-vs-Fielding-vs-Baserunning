from pybaseball import statcast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# purpose of project: among 4 factors of baseball (offense, pitching, defense, and baserunning) find the most accurate importance of each factor in terms of making 1 win

# load datasets
batting_df = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/Offense-Pitching-Fielding-and-Baserunning/data/FanGraphs Team Batting Data.csv')
pitching_df = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/Offense-Pitching-Fielding-and-Baserunning/data/FanGraphs Team Pitching Data.csv')



### 1. Data Cleaning ###
# check columns
print(batting_df.columns)
print(pitching_df.columns)

# change duplicate column names
batting_df.rename(columns={'R': 'RS'}, inplace=True)
batting_df.rename(columns={'WAR': 'bWAR'}, inplace=True)

pitching_df.rename(columns={'R': 'RA'}, inplace=True)
pitching_df.rename(columns={'BABIP': 'oppBABIP'}, inplace=True)
pitching_df.rename(columns={'EV': 'oppEV'}, inplace=True)
pitching_df.rename(columns={'WAR': 'pWAR'}, inplace=True)

# number of missing values
dfs = [batting_df, pitching_df]
for df in dfs:
    print('------- Missing Values -------')
    print(df.isnull().sum())

# replace missing values with predicted values using a linear regression model
no_obj = batting_df.select_dtypes(exclude='object')
imputer = IterativeImputer(random_state=1).fit_transform(no_obj)
batting_df = pd.DataFrame(data=imputer, columns=no_obj.columns)
print(batting_df.isnull().sum())

no_obj = pitching_df.select_dtypes(exclude='object')
imputer = IterativeImputer(random_state=1).fit_transform(no_obj)
pitching_df = pd.DataFrame(data=imputer, columns=no_obj.columns)
print(pitching_df.isnull().sum())

# number of duplicates
print('Number of Duplicates in Batting Data: {}'.format(batting_df.duplicated().sum()))
print('Number of Duplicates in Pitching Data: {}'.format(pitching_df.duplicated().sum()))

### 2. EDA ###
## 2-1, Batting Data EDA ##
# data structure
print('Data Structure: {}'.format(batting_df.shape))

# data types
print("------- Data Types -------")
print(batting_df.dtypes)

# descriptive summaries
print(batting_df.describe().to_string())

# correlation matrix
corrMatrix = round(batting_df.corr(), 2)
mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrMatrix, mask=mask, square=True, annot=True, annot_kws={'size': 8}, linewidths=.5,
            cbar_kws={'shrink': .5}, xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns, ax=ax)
ax.set_title('Correlation Matrix')

plt.show()

seasonal_Bdf = batting_df.groupby('Season')
seasonal_Bdf = seasonal_Bdf['wRC+'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(seasonal_Bdf['Season'], seasonal_Bdf['wRC+'], '-r')
ax.set_title('Yearly Changes in Team wRC+')
ax.set_xticks(range(1870, 2021, 10))
plt.show()