import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split 

pd.set_option("display.max_rows" , None)

#Read data
data = pd.read_csv('Data/Week 1/train.csv')

data.info()

#Remove player_id, num, type, ht
data['type'].value_counts()
data.drop(columns = ['type','player_id','num','ht'], inplace=True)
data.info()

#Check columns with missing values
missing = data.isna().sum()
missing.to_csv('missing')

#Rec rank has 39055 missing values, dunk ratio has 30793, pick 54705 -Way too many missing values. Decided to remove.
data.drop(columns = 'pick', inplace = True)
data.drop(columns = 'Rec_Rank', inplace = True)
data.drop(columns = 'dunks_ratio', inplace = True)
data.info()

#Exclude categorical values for now
df_no_cat = pd.DataFrame(data)
df_no_cat = df_no_cat.drop(columns=['team','conf','yr'])

#IMPUTE
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
knn_df = pd.DataFrame(imputer.fit_transform(df_no_cat),columns = df_no_cat.columns)

#Split into training/test
df = knn_df
features = df.drop(columns=['drafted'])
features.info()
target = df['drafted']

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size = 0.2, random_state=22)

from imblearn.over_sampling import SMOTE

#Balance dataset
smote = SMOTE(random_state=42)
feature_train_smote, target_train_smote = smote.fit_resample(feature_train, target_train)

len(feature_train_smote)
len(target_train_smote)
target_train_smote.value_counts()

#Scale data
#Minmax
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(feature_train_smote)
feature_train_scaled = scaler.transform(feature_train_smote)
feature_test_scaled = scaler.transform(feature_test)

feature_train_scaled = pd.DataFrame(feature_train_scaled)
feature_train_scaled.head(5)
feature_test_scaled = pd.DataFrame(feature_test_scaled)
feature_test_scaled.head(5)

#Logistic regression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter = 10000)
logistic_model.fit(feature_train_scaled, target_train_smote)
logistic_pred_smote = logistic_model.predict(feature_test_scaled)
roc_log_smote = roc_auc_score(target_test, logistic_pred_smote)
print(roc_log_smote)

#SVM
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(feature_train_scaled, target_train_smote)
svm_pred_smote = svm_model.predict(feature_test_scaled)
roc_svm_smote = roc_auc_score(target_test, svm_pred_smote)
print(roc_svm_smote)


#Tune HYPERPARAMETERS
from sklearn.model_selection import GridSearchCV

param_grid = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'C':[0.001, 0.01, 0.1, 1, 10, 100],
}

optimised = {
    'solver':['lbfgs'],
    'C':[100],
}

grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
grid_search.fit(feature_train_scaled, target_train_smote)
best_params = grid_search.best_params_
print(best_params)

grid_search = GridSearchCV(estimator=logistic_model, param_grid=optimised, scoring='roc_auc', cv=5, n_jobs=-1)
grid_search.fit(feature_train_scaled, target_train_smote)
best_params = grid_search.best_params_
print(best_params)



#Try parameters -0.95 roc
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter = 10000, solver='lbfgs', C=100)
logistic_model.fit(feature_train_scaled, target_train_smote)
logistic_pred_smote = logistic_model.predict(feature_test_scaled)
roc_log_smote = roc_auc_score(target_test, logistic_pred_smote)
print(roc_log_smote)


#Tune svm
params = {
    # 'C':[0.1,1,10,100],
    'kernel':['linear', 'poly','rbf']
}
svc = SVC()
grid_search = GridSearchCV(estimator=svc, param_grid=params, scoring='roc_auc', cv=5, n_jobs=-1)
grid_search.fit(feature_train_scaled, target_train_smote)
best_params = grid_search.best_params_
print(best_params)

feature_train_scaled.describe()

svc = SVC(kernel='rbf', random_state=44)
svc.fit(feature_train_scaled, target_train_smote)
svm_pred_smote = svc.predict(feature_test_scaled)
roc_svm_smote = roc_auc_score(target_test, svm_pred_smote)
print(roc_svm_smote)



#Try random forest
from sklearn.ensemble import RandomForestClassifier
rf_knn_balanced = RandomForestClassifier(random_state=42)
rf_knn_balanced.fit(feature_train_scaled, target_train_smote)
rf_knn_balanced_pred = rf_knn_balanced.predict(feature_test_scaled)
rf_knn_balanced_roc = roc_auc_score(target_test, rf_knn_balanced_pred)
print(rf_knn_balanced_roc)


#Try encoding categorical variables
cat = data[['yr','conf', 'team']]
cat['yr'].value_counts()
cat['conf'].value_counts()
cat['team'].value_counts()
cat.info()
data.info()

missing = data.isna().sum()
print(missing)

#Remove missing values from 'yr' 
data.dropna(subset = ['yr'],inplace=True)
missing = data.isna().sum()
print(missing)

#Remove incorrect values from 'yr'
values = ['0','57.1','42.9']
data = data[~data['yr'].isin(values)]
data['yr'].value_counts()

#Data encoded
data_encoded = pd.get_dummies(data, columns = ['conf','yr','team'],prefix ='encode')
data_encoded.info()
data_encoded['drafted'].value_counts()
#IMPUTE
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
knn_df = pd.DataFrame(imputer.fit_transform(data_encoded),columns=data_encoded.columns)
knn_df.head(3)

#Split into training/test
df = knn_df
features = df.drop(columns=['drafted'])
features.info()
target = df['drafted']

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size = 0.2, random_state=22)

from imblearn.over_sampling import SMOTE

#Balance dataset
smote = SMOTE(random_state=42)
feature_train_smote, target_train_smote = smote.fit_resample(feature_train, target_train)

len(feature_train_smote)
len(target_train_smote)
target_train_smote.value_counts()

#Scale data
#Minmax
feature_train_smote.head(3)
feature_train_smote_cols = feature_train_smote.columns.tolist()
print(feature_train_smote_cols)
feature_test_cols = feature_test.columns.tolist()
print(feature_test_cols)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(feature_train_smote)
feature_train_scaled = scaler.transform(feature_train_smote)
feature_train_scaled = pd.DataFrame(feature_train_scaled,columns = feature_train_smote_cols)
feature_train_scaled.head(3)
feature_test_scaled = scaler.transform(feature_test)
feature_test_scaled = pd.DataFrame(feature_test_scaled, columns = feature_test_cols)
feature_test_scaled.head(3)

#random forest
from sklearn.ensemble import RandomForestClassifier
rf_cats = RandomForestClassifier(random_state=42)
rf_cats.fit(feature_train_scaled, target_train_smote)
rf_cats_pred = rf_cats.predict(feature_test_scaled)
rf_cats_roc = roc_auc_score(target_test, rf_cats_pred)
print(rf_knn_balanced_roc)



logistic_model = LogisticRegression(max_iter = 10000)
logistic_model.fit(feature_train_scaled, target_train_smote)
logistic_pred_smote = logistic_model.predict(feature_test_scaled)
roc_log_smote = roc_auc_score(target_test, logistic_pred_smote)
print(roc_log_smote)


from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(feature_train_scaled, target_train_smote)
svm_pred_smote = svm_model.predict(feature_test_scaled)
roc_svm_smote = roc_auc_score(target_test, svm_pred_smote)
print(roc_svm_smote)

feature_train_scaled.head(3)

importances = rf_cats.feature_importances_
print(importances)


feature_names = pd.DataFrame(feature_train_scaled).columns.to_list()
feature_train_scaled.head(3)
print(feature_names)
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame by importance in descending order to see the most important features first
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print or display the sorted DataFrame
print(feature_importance_df)
