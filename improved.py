import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

missing_val_count_by_column = (X_train.isnull().sum())

num_rows = X_train.shape[0]
num_cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].shape[0]
tot_missing = missing_val_count_by_column.sum()


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#Preliminary Investigation
# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

#Shape of training data, num of rows
num_rows = X_train.shape[0]
#Num of columns with missing values
num_cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].shape[0]
#Num of Missing entries in total
tot_missing = missing_val_count_by_column.sum()

#STEP 2, drop columns with missing values
#names of missingData columns
cols_missing_data = [col for col in X_train.columns 
                        if X_train[col].isnull().any()] 
#drop the said columns from X_train and X_valid
reduced_X_train = X_train.drop(cols_missing_data, axis=1) 
reduced_X_valid = X_valid.drop(cols_missing_data, axis=1)

#Check deletion MAE Score
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

#Step 3 imputation (instead of deleting values)
#Impute X_train and Valid 
imp = SimpleImputer()
imputed_X_train = pd.DataFrame(imp.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imp.transform(X_valid))

# Put back column names
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

#Check MAE score
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))


#Generate test predictions
#New imputer with median strategy
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

#preprocessing test data

final_X_test = pd.DataFrame(final_imputer.transform(X_test)) 
final_X_test.columns = X_test.columns 
preds_test =  model.predict(final_X_test) 

