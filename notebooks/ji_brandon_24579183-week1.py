import pandas as pd
pd.set_option("display.max_columns" , None)
pd.set_option("display.max_rows",None)

#Read data
data = pd.read_csv("/Week%201/train.csv?token=GHSAT0AAAAAACGH327SAEMLUN76VMNLCSZUZHJ24MQ")

#Drop irrelevant columns
data = data.drop(columns = ["player_id", "num", "type"])
data = data.drop(columns = ["team", "conf", "ht"])
data.info()
data['drafted'].value_counts()
 
#Check missing values
data.isnull().sum()

#Drop columns that contain >100 missing values.
data_wo_na = data.drop(columns = ["rimmade", "rimmade_rimmiss", "midmade", "midmade_midmiss", "mid_ratio", "dunksmade","dunksmiss_dunksmade","dunks_ratio","pick", "Rec_Rank", "ast_tov", "rim_ratio", "yr"])
data_wo_na.isna().sum()

#Drop all rows that contain missing values
len(data_wo_na)
data_no_na = data_wo_na.dropna()
data_no_na.isna().sum()
len(data_no_na)

#Check categorical values (team, conf, ht, type)
data_no_na['team'].nunique()
data_no_na['conf'].nunique()
data_no_na['ht'].nunique()
data_no_na['type'].nunique()

#Remove categorical values
data_removed = data_no_na

### Modelling ###

from sklearn.tree import DecisionTreeClassifier #Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#Split into training and test set
df = data_removed
features = df.drop(columns=['drafted'])
features.info()
target = df['drafted']

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size = 0.2, random_state=22)
len(df)
len(feature_train)
len(feature_test)
len(target_train)
len(target_test)
feature_train.info()
feature_test.info()
target_train.info()
target_test.info()

#DecisionTree
clf = DecisionTreeClassifier()
clf = clf.fit(feature_train, target_train)
ypred = clf.predict(feature_test)
ypred = pd.Series(ypred)

#Calculate eval
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(target_test, ypred)

precision = precision_score(target_test, ypred)

recall = recall_score(target_test, ypred)

f1 = f1_score(target_test, ypred)

roc_auc = roc_auc_score(target_test, ypred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)

#Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rfclass = rf.fit(feature_train, target_train)
y_pred = rfclass.predict(feature_test)

#Evaluate

accuracy = accuracy_score(target_test, y_pred)

precision = precision_score(target_test, y_pred)

recall = recall_score(target_test, y_pred)

f1 = f1_score(target_test, y_pred)

roc_auc = roc_auc_score(target_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)

#Tune hyper parameters
from sklearn.model_selection import RandomizedSearchCV, train_test_split

param_dist = {'n_estimators': [50, 100, 200, 300, 400 ,500], 'max_depth': [None, 10, 20, 30], 'max_features':["sqrt", "log2", None]}
randomforest = RandomForestClassifier()
rand_search = RandomizedSearchCV(randomforest, param_distributions = param_dist, n_iter=5, cv=5, random_state=45)

rand_search.fit(feature_train, target_train)

best_rf = rand_search.best_estimator_
print(best_rf)
#Use optimised parameters
rf = RandomForestClassifier(n_estimators= 400, max_depth = 10, max_features="log2")
optimisedrf = rf.fit(feature_train, target_train)
optimised_pred = optimisedrf.predict(feature_test)

accuracy = accuracy_score(target_test, optimised_pred)

precision = precision_score(target_test, optimised_pred)

recall = recall_score(target_test, optimised_pred)

f1 = f1_score(target_test, optimised_pred)

roc_auc = roc_auc_score(target_test, optimised_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)


#Tune decision tree hyper parameters
dt = DecisionTreeClassifier()
param_dt = {'max_depth': [None, 10, 20, 30], 'max_features':["sqrt", "log2", None], 'min_samples_split': [2,5,10,15,20]}
optimise_dt = RandomizedSearchCV(dt, param_distributions=param_dt, n_iter=5, cv = 5, random_state=55)
optimise_dt.fit(feature_train,target_train)
best_dt = optimise_dt.best_params_
print(best_dt)

optimiseddt = DecisionTreeClassifier(min_samples_split=20, max_features="sqrt", max_depth=10)
dt2 = optimiseddt.fit(feature_train, target_train)
pred = dt2.predict(feature_test)

accuracy = accuracy_score(target_test, pred)

precision = precision_score(target_test, pred)

recall = recall_score(target_test, pred)

f1 = f1_score(target_test, pred)

roc_auc = roc_auc_score(target_test, pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)
