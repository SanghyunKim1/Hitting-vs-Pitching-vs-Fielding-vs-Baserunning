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
from statsmodels.tsa.seasonal import seasonal_decompose
from math import sqrt
import statsmodels.api as sm
import missingno as msno
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

# missing data visualization
msno.matrix(team_df)
msno.heatmap(team_df)

# drop variables that is not worth imputing
team_df.drop(['EV', 'oppEV'], axis=1, inplace=True)

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
# for more information about how and why I selected features below, please refer to README.md on GitHub
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
# as all the independent variables have different ranges, scale them

# feature scaling
no_scale = ['Season', 'Team', 'wPCT']
scale = team_df.drop(no_scale, axis=1)
cols = list(scale.columns)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(scale)
scaled_df = pd.DataFrame(scaled_data, columns=cols)

scaled_df = pd.concat([team_df[no_scale], scaled_df], axis=1)

# KDE plot
fig, ax = plt.subplots(figsize=(8, 8))

for col in cols:
    sns.kdeplot(scaled_df[col], ax=ax, label=col)
    ax.set_title('After StandardScaler')
    ax.set_xlabel('Data Scale')
    plt.legend(loc=1)
plt.show()

# time series plot
ind_vars = ['wRC+', 'wOBA', 'WHIP', 'xFIP', 'Def', 'BsR']
season_df = scaled_df.groupby('Season')[ind_vars].median()

fig, axes = plt.subplots(3, 2, figsize=(18, 10))
palette = plt.get_cmap('Set1')
num = 0

for col, ax in zip(season_df, axes.flatten()[:7]):
    num += 1
    for var in season_df:
        ax.plot(season_df[var], marker='', color='grey', linewidth=.6, alpha=.3)

    ax.plot(season_df[col], marker='', color=palette(num), linewidth=2.4, alpha=.9,
             label=col)

    ax.set_title(col, loc='center', fontsize=12, fontweight=0)
    ax.set_xticks(range(1870, 2020, 10))

plt.suptitle('Changes in Each Statistic through The MLB History',
             fontsize=15, fontweight=1, y=0.95)
plt.text(0.5, 0.02, 'Year', ha='center', va='center')
plt.text(0.06, 0.5, 'Scale', ha='center', va='center', rotation='vertical')
plt.show()

# create year bins
bins = [1870, 1900, 1920, 1940, 1960, 1980, 2000, 2020]
labels = ['1871-1899', '1900-1919', '1920-1939', '1940-1959',
          '1960-1979', '1980-1999', '2000-2019']

team_df['Year Band'] = pd.cut(team_df['Season'], bins, labels=labels,
                                include_lowest=True, right=False)
scaled_df['Year Band'] = pd.cut(scaled_df['Season'], bins, labels=labels,
                                include_lowest=True, right=False)


# changes in each statistic measured by 'mean' and 'median' through the periods
generation = team_df.groupby('Year Band', as_index=False)
result = generation[ind_vars].agg(['mean', 'median'])
print(result.to_string())

# compare teams whose 'wPCT' is higher than 0.500 with teams whose 'wPCT' is less than 0.500
# note: scaled data is used for this analysis to accurately compare all the different stats
scaled_df['wPCT > 0.500'] = np.where(scaled_df['wPCT'] >= 0.500, '> 0.500', '< 0.500')

five_hundered = scaled_df.groupby('wPCT > 0.500')
team_stat = five_hundered[ind_vars].agg(['mean', 'median'])

print('------- Winning Team Stats -------')
print(team_stat.to_string())



# offense analysis


# # correlation matrix
# corrMatrix = team_df.corr()
# print(corrMatrix.to_string())
#
# mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
#
# fig, ax = plt.subplots(figsize=(10, 10))
#
# sns.heatmap(corrMatrix, mask=mask, annot=True, annot_kws={'size':10}, square=True, linewidths=.5,
#             cbar_kws={'shrink': .5}, xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns, ax=ax)
# ax.set_title('Correlation Matrix')
#
# plt.show()
#
# x = team_df.drop(['Season', 'Team', 'wPCT'], axis=1)
# y = team_df['wPCT']
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
# x = team_df.drop(['Season', 'Team', 'wPCT'], axis=1)
# x = sm.add_constant(x)
# y = team_df['wPCT']
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
# x = team_df.drop(['Season', 'Team', 'wPCT'], axis=1)
# y = team_df['wPCT']
#
# model = LinearRegression()
# cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
# cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
# cv_rmse = np.sqrt(-1 * cv_mse)
#
# print(cv_r2.mean())
# print(cv_rmse.mean())