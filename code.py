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
import statsmodels.api as sm
from math import sqrt
from scipy import stats
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# purpose of project: among 4 factors of baseball (hitting, pitching, defense, and baserunning) find the most accurate importance of each factor in terms of making 1 win

# load datasets
batting_df = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/data/FanGraphs Team Batting Data.csv')
pitching_df = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning/data/FanGraphs Team Pitching Data.csv')


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

# number of missing values
print('------- Missing Data -------')
print(team_df.isnull().sum())

# missing data visualization
msno.matrix(team_df)

missing_df = pd.DataFrame(index=list(team_df.columns))
missing_df['% Missing Data'] = (team_df.isnull().sum() / len(team_df))
missing_df['% Missing Data'] = missing_df['% Missing Data'].map(lambda x: "{0:.2f}%".format(x*100))
missing_vars = missing_df.loc[missing_df['% Missing Data'] != '0.00%']
print(missing_vars.sort_values('% Missing Data', ascending=False))

# drop variables that is not worth imputing
team_df.drop(['EV', 'oppEV'], axis=1, inplace=True)

# replace missing values with predicted values using a linear regression model
no_obj = team_df.select_dtypes(exclude='object')
imputer = IterativeImputer(random_state=1).fit_transform(no_obj)
impute_df = pd.DataFrame(data=imputer, columns=no_obj.columns)

team_df = pd.concat([team_df['Team'], impute_df], axis=1)

print('------- Missing Data -------')
print(team_df.isnull().sum())

# number of duplicates
print('Number of Duplicates: {}'.format(team_df.duplicated().sum()))

# note: based on domain knowledge, I filtered the best features for this analysis
# for more information about how and why I selected features below,
# please refer to https://github.com/shk204105/Hitting-vs-Pitching-vs-Fielding-vs-Baserunning

# drop unnecessary variables
vars_keep = ['Season', 'Team', 'wPCT', 'wOBA', 'FIP', 'Def', 'BsR']
team_df = team_df[vars_keep]



### 2. EDA ###
# data structure
print('Data Structure: {}'.format(team_df.shape))

# normality
# 'wPCT' normality
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.histplot(team_df['wPCT'], kde=True, ax=axes[0])
axes[0].set_title('Team Winning Percentage Histogram')

stats.probplot(team_df['wPCT'], plot=axes[1])
axes[1].set_title('Team Winning Percentage Q-Q Plot')

plt.show()

# independent variables normality
ind_vars = ['wOBA', 'FIP', 'Def', 'BsR']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for col, ax in zip(ind_vars, axes.flatten()[:7]):
    sns.histplot(team_df[col], kde=True, color='navy', ax=ax)
    ax.set_title('Team {} Histogram'.format(col))

plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for col, ax in zip(ind_vars, axes.flatten()[:7]):
    stats.probplot(team_df[col], plot=ax)
    ax.set_title('Team {} Q-Q Plot'.format(col))

plt.show()

# descriptive summaries
print(team_df.describe().to_string())
# as all the independent variables have different ranges, scale them

# feature scaling
not_scale = ['Season', 'Team', 'wPCT']
for_scale = team_df.drop(not_scale, axis=1)
cols = list(for_scale.columns)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(for_scale)
scaled_df = pd.DataFrame(scaled_data, columns=cols)

scaled_df = pd.concat([team_df[not_scale], scaled_df], axis=1)

# KDE plot
fig, ax = plt.subplots(figsize=(10, 10))

for col in cols:
    sns.kdeplot(scaled_df[col], ax=ax, label=col)
    ax.set_title('After StandardScaler')
    ax.set_xlabel('Data Scale')
    plt.legend(loc=1)

plt.show()

# Changes in median values of each stat throughout the MLB history
# note: scaled data is used for this analysis to accurately compare all the different stats
season_df = scaled_df.groupby('Season')[ind_vars].median()

fig, axes = plt.subplots(2, 2, figsize=(18, 7))
palette = plt.get_cmap('Set1')
num = 0

for col, ax in zip(season_df, axes.flatten()[:7]):
    num += 1
    for var in season_df:
        ax.plot(season_df[var], marker='', color='grey', linewidth=.6, alpha=.3)

    ax.plot(season_df[col], marker='', color=palette(num), linewidth=1.8, alpha=.9,
             label=col)

    ax.set_title(col, loc='center', fontsize=12, fontweight=0)
    ax.set_xticks(range(1870, 2020, 10))

plt.suptitle('Changes in Median Values of Each Stat throughout The MLB History',
             fontsize=15, fontweight=1, y=0.95)
plt.show()

# create year bins
bins = [1870, 1900, 1920, 1940, 1960, 1980, 2000, 2020]
labels = ['1871-1899', '1900-1919', '1920-1939', '1940-1959',
          '1960-1979', '1980-1999', '2000-2019']
data = [team_df, scaled_df]

for df in data:
    df['Era'] = pd.cut(df['Season'], bins, labels=labels,
                       include_lowest=True, right=False)

# changes in 'mean' and 'median' stats in different eras
era_grouped = team_df.groupby('Era', as_index=False)
era_grouped_stats = era_grouped[ind_vars].agg(['mean', 'median'])
print(era_grouped_stats.to_string())

# Changes in median values of each stat throughout different eras
scaled_era_df = scaled_df.groupby('Era')
scaled_era_stats = scaled_era_df[ind_vars].median()

fig, axes = plt.subplots(2, 2, figsize=(18, 7))
palette = plt.get_cmap('Set1')
num = 0

for col, ax in zip(ind_vars, axes.flatten()[:7]):
    num += 1
    for var in scaled_era_stats:
        ax.plot(scaled_era_stats[var], marker='', color='grey', linewidth=.6, alpha=.3)

    ax.plot(scaled_era_stats[col], marker='', color=palette(num), linewidth=1.8, alpha=.9)
    ax.set_title(col, loc='center', fontsize=12, fontweight=0)

plt.suptitle('Changes in Each Median Stat throughout Different Eras')
plt.show()

# compare teams whose 'wPCT' is higher than 0.500 with teams whose 'wPCT' is less than 0.500
scaled_df['wPCT > 0.500'] = np.where(scaled_df['wPCT'] >= 0.500, '> 0.500', '< 0.500')

wPCT_grouped = scaled_df.groupby('wPCT > 0.500')
wPCT_grouped_stats = wPCT_grouped[ind_vars].median().reset_index()
print(wPCT_grouped_stats)

# grouped bar plot
barWidth = 0.15
wOBA = wPCT_grouped['wOBA'].median()
FIP = wPCT_grouped['FIP'].median()
Def = wPCT_grouped['Def'].median()
BsR = wPCT_grouped['BsR'].median()

category = [wOBA, FIP, Def, BsR]

r1 = np.arange(len(wOBA))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

rs = [r1, r2, r3, r4]

category_color = plt.get_cmap('Pastel1')
num = 0

for var, r, col in zip(category, rs, ind_vars):
    num += 1
    plt.bar(r, var, color=category_color(num), width=barWidth, edgecolor='white', label=col)

plt.title('Stat Comparison based on Team Winning Percentage')
plt.xlabel('wPCT', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(wOBA))], ['Lower than 0.500', 'Higher than 0.500'])
plt.ylabel('Scale', fontweight='bold')
plt.legend(loc='lower right')

plt.show()

# correlation matrix
corrMatrix = team_df.corr()
print('------- Correlation -------')
print(corrMatrix.to_string())

mask = np.triu(np.ones_like(corrMatrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corrMatrix, mask=mask, annot=True, annot_kws={'size':10}, square=True, cmap='plasma', linewidths=.5,
            cbar_kws={'shrink': .5}, xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns, ax=ax)
ax.set_title('Correlation Matrix')

plt.show()

# scatter plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for col, ax in zip(ind_vars, axes.flatten()[:7]):
    sns.regplot(col, 'wPCT', data=team_df, scatter_kws={'color':'black'},
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

fig, ax = plt.subplots()

sns.lineplot('n Estimators', 'Score', data=scores_df, color='red', ax=ax)
ax.set_title('Changes in Scores given The Number of Estimators')
ax.set_xlabel('Number of Estimators')
ax.set_ylabel('Score')

plt.show()

Nbest_estimators = scores_df[scores_df['Score'] == scores_df['Score'].max()]
print('------- Best Number of Estimators -------')
print(Nbest_estimators)

Nbest_estimators = scores_df.loc[scores_df['Score'] == scores_df['Score'].max(), 'n Estimators'].item()

# random forest regression
model = RandomForestRegressor(n_estimators=Nbest_estimators, n_jobs=-1, random_state=0)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
mse = metrics.mean_squared_error(y_test, y_predict)

print('------- Random Forest Regression (1871-2019) -------')
print('R-squared: {}'.format(score))
print('RMSE: {}'.format(sqrt(mse)))

# random forest feature importance
importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]

print('------- Random Forest Feature Importance (1871-2019) -------')
for i in sorted_idx:
    print('{} Imoportance: {}'.format(x.columns[i], round(importance[i], 3)))



### 5. Cross-era Comparison ###
eras = sorted(list(team_df['Era'].unique()))
importance_dict = {}

for era in eras:
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


    # random forest feature importance cross-era comparison
    importance = model.feature_importances_
    importance_dict[era] = list(np.round(importance, 3))
    sorted_indices = np.argsort(importance)[::-1]

    print('------- Random Forest Feature Importance ({}) -------'.format(era))
    for i in sorted_indices:
        print('{} Importance: {}'.format(x.columns[i], round(importance[i], 3)))

    # # permutation importance cross-era comparison
    # r = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=0)
    # sorted_idx = r.importances_mean.argsort()
    #
    # print('------- Permutation Importance ({})-------'.format(era))
    # for i in sorted_idx[::-1]:
    #     print('{} Importance: {}'.format(x_test.columns[i], round(r.importances_mean[i], 3)))
    #
    # plt.barh(x_test.columns[sorted_idx], r.importances_mean[sorted_idx])
    # plt.title('Permutation Importance ({})'.format(era))
    # plt.xlabel('Importance')
    # plt.show()

    # K-fold cross validation
    cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
    cv_rmse = np.sqrt(-1 * cv_mse)

    print('------- 10-Fold Cross Validation ({}) -------'.format(era))
    print('Mean RMSE: {}'.format(cv_rmse.mean()))


### 6. Feature Importance Visualization ###
category_names = ind_vars
results = dict(sorted(importance_dict.items()))

def importance_score(results, category_names):

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('viridis')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(float(c)), ha='center', va='center',
                    color=text_color, fontsize='small')
    ax.set_title('Changes in Impacts of Stats on Team Winning Percentage',
                 fontweight='bold')
    ax.set_ylabel('Era', fontweight='bold')
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, -0.1),
              loc='lower left', fontsize='medium')

    return fig, ax

importance_score(results, category_names)
plt.show()



### 7. Multiple Linear Regression ###
def linear_model(era):
    data = scaled_df[scaled_df['Era'] == era]
    x = data.drop(['Season', 'Team', 'wPCT', 'Era', 'wPCT > 0.500'], axis=1)
    x = sm.add_constant(x)
    y = data['wPCT']

    lm = sm.OLS(y, x).fit()

    print('------- Linear Regression Result ({}) -------'.format(era))
    print(lm.summary())

    # residual plot
    fitted_y = lm.fittedvalues

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.residplot(fitted_y, 'wPCT', data=data, lowess=True, scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, ax=ax)
    ax.set_title('Residuals vs Fitted Values ({})'.format(era))
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel("Residuals")

    plt.show()

linear_model('1871-1899')
linear_model('1900-1919')
linear_model('1920-1939')
linear_model('1940-1959')
linear_model('1960-1979')
linear_model('1980-1999')
linear_model('2000-2019')