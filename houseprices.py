import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice
feature_names = ['LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd']
X = home_data[feature_names]



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

iowaPred = iowa_model.predict(X)
val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_y, val_predictions)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

smallestError = 30000
bestMLN = 71
#for mln in range(50,250):
#   my_mae = get_mae(mln, train_X, val_X, train_y, val_y)
#    if my_mae < smallestError:
#        smallestError = my_mae
#        bestMLN = mln

final_model = DecisionTreeRegressor(max_leaf_nodes=bestMLN)
final_model.fit(X, y)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_pred = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_pred,val_y)

