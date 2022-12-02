import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


# Read the data
train_data = pd.read_csv('pipelines/train.csv', index_col='Id')
test_data = pd.read_csv('pipelines/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

#pipeline
#Remember to do preprocessing steps and add model
my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# crossvalScore * -1 because negative MAE because scikitlearn is weird
#include pipeline, Training and result data
#cv = 5, 5 folds
#scoring is -MAE

scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())

def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    tempPipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators= n_estimators, random_state=0))
    ])

    scores = -1 * cross_val_score(tempPipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()


#You can loop through, changing number of estimators to find the best number

#you can plot stuff too!
# Assume results is a dictionary of key (n-estimators)
# and values are the MAE of get_score(key)
import matplotlib.pyplot as plt
plt.plot(list(results.keys()), list(results.values()))
plt.show()
