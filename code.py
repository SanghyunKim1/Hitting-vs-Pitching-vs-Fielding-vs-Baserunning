import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from math import sqrt
from scipy import stats
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

# normality
# 'wPCT' normality
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.histplot(team_df['wPCT'], kde=True, ax=axes[0])
axes[0].set_title('Team Winning Percentage Histogram')

stats.probplot(team_df['wPCT'], plot=axes[1])
axes[1].set_title('Team Winning Percentage Q-Q Plot')

plt.show()

# independent variables normality
ind_vars = ['wRC+', 'wOBA', 'WHIP', 'xFIP', 'Def', 'BsR']

fig, axes = plt.subplots(3, 2, figsize=(15, 15))

for col, ax in zip(ind_vars, axes.flatten()[:7]):
    sns.histplot(team_df[col], kde=True, color='navy', ax=ax)
    ax.set_title('Team {} Histogram'.format(col))

plt.show()

fig, axes = plt.subplots(3, 2, figsize=(15, 15))

for col, ax in zip(ind_vars, axes.flatten()[:7]):
    stats.probplot(team_df[col], plot=ax)
    ax.set_title('Team {} Q-Q Plot'.format(col))

plt.show()

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
plt.show()

# create year bins
bins = [1870, 1900, 1920, 1940, 1960, 1980, 2000, 2020]
labels = ['1871-1899', '1900-1919', '1920-1939', '1940-1959',
          '1960-1979', '1980-1999', '2000-2019']

team_df['Era'] = pd.cut(team_df['Season'], bins, labels=labels,
                        include_lowest=True, right=False)
scaled_df['Era'] = pd.cut(scaled_df['Season'], bins, labels=labels,
                          include_lowest=True, right=False)

# changes in 'mean' and 'median' statistics measured in each era
era_grouped = team_df.groupby('Era', as_index=False)
era_grouped_stats = era_grouped[ind_vars].agg(['mean', 'median'])
print(era_grouped_stats.to_string())

# visualize changes in each statistic through different eras
scaled_era_df = scaled_df.groupby('Era')
scaled_era_stats = scaled_era_df[ind_vars].median()

fig, axes = plt.subplots(3, 2, figsize=(18, 10))
palette = plt.get_cmap('Set1')
num = 0

for col, ax in zip(ind_vars, axes.flatten()[:7]):
    num += 1
    for var in scaled_era_stats:
        ax.plot(scaled_era_stats[var], marker='', color='grey', linewidth=.6, alpha=.3)

    ax.plot(scaled_era_stats[col], marker='', color=palette(num), linewidth=2.4, alpha=.9)
    ax.set_title(col, loc='center', fontsize=12, fontweight=0)

plt.suptitle('Changes in Each Statistic through Different Eras')
plt.show()

# compare teams whose 'wPCT' is higher than 0.500 with teams whose 'wPCT' is less than 0.500
# note: scaled data is used for this analysis to accurately compare all the different stats
scaled_df['wPCT > 0.500'] = np.where(scaled_df['wPCT'] >= 0.500, '> 0.500', '< 0.500')

wPCT_grouped = scaled_df.groupby('wPCT > 0.500')
wPCT_grouped_stats = wPCT_grouped[ind_vars].median().reset_index()
print(wPCT_grouped_stats)

# grouped bar plot
barWidth = 0.15
wRC_plus = wPCT_grouped['wRC+'].median()
wOBA = wPCT_grouped['wOBA'].median()
WHIP = wPCT_grouped['WHIP'].median()
xFIP = wPCT_grouped['xFIP'].median()
Def = wPCT_grouped['Def'].median()
BsR = wPCT_grouped['BsR'].median()

category = [wRC_plus, wOBA, WHIP, xFIP, Def, BsR]

r1 = np.arange(len(wRC_plus))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]

rs = [r1, r2, r3, r4, r5, r6]

category_color = plt.get_cmap('Pastel1')
num = 0

for var, r, col in zip(category, rs, ind_vars):
    num += 1
    plt.bar(r, var, color=category_color(num), width=barWidth, edgecolor='white', label=col)

plt.title('Stat Comparison based on Team Winning Percentage')
plt.xlabel('wPCT', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(wRC_plus))], ['Lower than 0.500', 'Higher than 0.500'],
           ha='left')
plt.ylabel('Scale', fontweight='bold')
plt.legend()

plt.show()


# correlation matrix
corrMatrix = team_df.corr()
print('------- Correlation -------')
print(corrMatrix.to_string())

mask = np.triu(np.ones_like(corrMatrix, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrMatrix, mask=mask, annot=True, annot_kws={'size':10}, square=True, cmap='plasma', linewidths=.5,
            cbar_kws={'shrink': .5}, xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns, ax=ax)
ax.set_title('Correlation Matrix')

plt.show()

# scatter plots
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

for col, ax in zip(ind_vars, axes.flatten()[:7]):
    sns.regplot(col, 'wPCT', data=scaled_df, scatter_kws={'color':'black'},
                line_kws={'color':'red'}, ax=ax)
    ax.set_title('Correlation between Team {} and Winning Percentage'.format(col))

plt.show()



### 3. Random Forest Regression ###
# find the best number of estimators for random forest regression
x = team_df.drop(['Season', 'Team', 'wPCT', 'Era'], axis=1)
y = team_df['wPCT']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = RandomForestRegressor(n_jobs=-1, random_state=1)

n_estimators = np.arange(10, 200, 10)
scores = []

for n in n_estimators:
    model.set_params(n_estimators=n)
    model.fit(x_train, y_train)
    scores.append([n, model.score(x_test, y_test)])

scores_df = pd.DataFrame(scores, columns=['n Estimators', 'Score'])
print(scores_df)

fig, ax = plt.subplots()

sns.lineplot('n Estimators', 'Score', data=scores_df, color='red', ax=ax)
ax.set_title('Changes in Scores given The Number of Estimators')
ax.set_xlabel('Number of Estimators')
ax.set_ylabel('Score')

plt.show()

Nbest_estimators = scores_df[scores_df['Score'] == scores_df['Score'].max()]
print('------- Best Number of Estimators -------')
print(Nbest_estimators)

# random forest regression
model = RandomForestRegressor(n_estimators=40, n_jobs=-1, random_state=0)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
mse = metrics.mean_squared_error(y_test, y_predict)

print('------- Random Forest Result -------')
print('R-squared: {}'.format(score))
print('RMSE: {}'.format(sqrt(mse)))



### 4. permutation importance ###
r = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=0)
sorted_idx = r.importances_mean.argsort()

plt.barh(x_test.columns[sorted_idx], r.importances_mean[sorted_idx])
plt.title('Permutation Importance (1871-2019)')
plt.xlabel('Importance')

plt.show()

print('------- Permutation Importance -------')
for i in sorted_idx[::-1]:
    print('{} Importance: {}'.format(x_test.columns[i], round(r.importances_mean[i], 3)))



### 5. Cross-era Comparison ###
def random_forest(era):
    data = team_df[team_df['Era'] == era]
    x = data.drop(['Season', 'Team', 'wPCT', 'Era'], axis=1)
    y = data['wPCT']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = RandomForestRegressor(n_jobs=-1, random_state=1)
    model.fit(x_train, y_train)

    n_estimators = list(range(10, 200, 10))
    scores = []

    for n in n_estimators:
        model.set_params(n_estimators=n)
        model.fit(x_train, y_train)
        scores.append([n, model.score(x_test, y_test)])

    scores_df = pd.DataFrame(scores, columns=['n Estimators', 'Score'])
    Nbest_estimators = scores_df.loc[scores_df['Score'] == scores_df['Score'].max(), 'n Estimators'].item()

    model = RandomForestRegressor(n_estimators=Nbest_estimators, n_jobs=-1, random_state=0)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    score = model.score(x_test, y_test)
    mse = metrics.mean_squared_error(y_test, y_predict)

    print('------- Random Forest Regression ({}) -------'.format(era))
    print('R-squared: {}'.format(score))
    print('RMSE: {}'.format(sqrt(mse)))

    # permutation importance cross-era comparison
    r = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=0)
    sorted_idx = r.importances_mean.argsort()

    print('------- Permutation Importance ({})-------'.format(era))
    for i in sorted_idx[::-1]:
        print('{} Importance: {}'.format(x_test.columns[i], round(r.importances_mean[i], 3)))

    plt.barh(x_test.columns[sorted_idx], r.importances_mean[sorted_idx])
    plt.title('Permutation Importance ({})'.format(era))
    plt.xlabel('Importance')
    plt.show()


random_forest('1871-1899')
random_forest('1900-1919')
random_forest('1920-1939')
random_forest('1940-1959')
random_forest('1960-1979')
random_forest('1980-1999')
random_forest('2000-2019')

