import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from math import sqrt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
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

# merge datasets
team_df = pd.merge(batting_df, pitching_df, on=['Season', 'Team'], how='outer')

# create new columns
team_df['UER'] = team_df['RA'] - team_df['ER']
team_df['G'] = team_df['W'] + team_df['L']
team_df['wPCT'] = round((team_df['W']/team_df['G']), 3)

# check data types
print(team_df.dtypes)

# categorical variables
obj_cols = list(team_df.select_dtypes(include='object'))
print(team_df[obj_cols].head())

# number of missing values
print('------- Missing Values -------')
print(team_df.isnull().sum())

# replace missing values with predicted values using a linear regression model
no_obj = team_df.select_dtypes(exclude='object')
imputer = IterativeImputer(random_state=1).fit_transform(no_obj)
impute_df = pd.DataFrame(data=imputer, columns=no_obj.columns)

team_df = pd.concat([team_df['Team'], impute_df], axis=1)

print('------- Missing Values -------')
print(team_df.isnull().sum())

# number of duplicates
print('Number of Duplicates: {}'.format(team_df.duplicated().sum()))

# note: based on domain knowledge, I filtered the best features for this analysis
# for more information, please refer to README.md on GitHub
# drop unnecessary variables
vars_keep = ['Season', 'Team', 'wPCT', 'wRC+', 'wOBA', 'WHIP', 'xFIP', 'Def', 'BsR']
team_df = team_df[vars_keep]



### 2. EDA ###
# data structure
print('Data Structure: {}'.format(team_df.shape))

# change data types
team_df['Season'] = team_df['Season'].astype('int')

# descriptive summaries
print(team_df.describe().to_string())

# correlation matrix
corrMatrix = team_df.corr()
print(corrMatrix.to_string())

mask = np.triu(np.ones_like(corrMatrix, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrMatrix, mask=mask, annot=True, annot_kws={'size':10}, square=True, linewidths=.5,
            cbar_kws={'shrink': .5}, xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns, ax=ax)
ax.set_title('Correlation Matrix')

plt.show()





# # feature scaling
# data = team_df[['wPCT', 'wRC+', 'wOBA', 'WHIP', 'xFIP', 'Def', 'BsR']]
# scale = data.iloc[:, data.columns != 'wPCT']
# cols = list(scale.columns)
#
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(scale)
# scaled_df = pd.DataFrame(scaled_data, columns=cols)
#
# data = pd.concat([data['wPCT'], scaled_df], axis=1)
#
# x = data.iloc[:, data.columns != 'wPCT']
# y = data['wPCT']
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
#
# lm = linear_model.LinearRegression().fit(x_train, y_train)
# y_predict = lm.predict(x_test)
#
# print(lm.intercept_)
# print(lm.coef_)
# print(sqrt(metrics.mean_squared_error(y_test, y_predict)))
# print(metrics.r2_score(y_test, y_predict))
#
# x = data.iloc[:, data.columns != 'wPCT']
# x = sm.add_constant(x)
# y = data['wPCT']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
#
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# x = data.iloc[:, data.columns != 'wPCT']
# y = data['wPCT']
#
# model = LinearRegression()
# cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
# cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
# cv_rmse = np.sqrt(-1 * cv_mse)
#
# print(cv_r2.mean())
# print(cv_rmse.mean())