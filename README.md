# Medals-Winning-Prediction
#Predict how many medals each country will win in the olympics using a linear regression model.

import pandas as pd
teams = pd.read_csv('teams.csv')
teams
teams = teams[["team","country","year","athletes","age","prev_medals","medals"]]
teams

 #Select all numeric features, including 'medals'
all_numeric_features = ["year", "athletes", "age", "prev_medals", "medals"] 

# Calculate the correlation matrix for all numeric features
correlation_matrix = teams[all_numeric_features].corr()

# To specifically see the correlation of predictors with 'medals':
correlations_with_medals = correlation_matrix["medals"]
print(correlations_with_medals)

import seaborn as sns
sns.lmplot(x="athletes",y="medals",data=teams, fit_reg=True, ci=None)

sns.lmplot(x="age",y="medals",data=teams,fit_reg=True, ci=None)

teams.plot.hist(y="medals")

teams[teams.isnull().any(axis=1)]

teams = teams.dropna()
teams

train = teams[teams["year"]< 2012].copy()
test = teams[teams["year"]>=2012].copy()

train.shape

test.shape

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
features=["athletes","prev_medals"]
target = "medals"
reg.fit(train[features],train[target])
LinearRegression()
predictions = reg.predict(test[features])
predictions

test["predictions"] = predictions  # Add predictions as a column to the DataFrame
test.loc[test["predictions"]< 0,"predictions"] = 0

test["predictions"]= test["predictions"].round()
test


from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(test["medals"], test["predictions"])
error

teams.describe()["medals"]

test[test["team"]=="USA"]

test[test["team"]=="IND"]

error = (test["medals"]-test["predictions"]).abs()
error

error_by_team = error.groupby(test["team"]).mean()
error_by_team

medals_by_team = test["medals"].groupby(test["team"]).mean()
medals_by_team

error_ratio = error_by_team/medals_by_team
error_ratio

error_ratio[-pd.isnull(error_ratio)]

#To remove infinit values
import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio

error_ratio.plot.hist()

error_ratio.sort_values()
