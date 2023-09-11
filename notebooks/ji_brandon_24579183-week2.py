import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split 

pd.set_option("display.max_columns" , None)

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

#Check outliers
outlier_counts = {}
for column in df_no_cat:
    df_no_cat[column] = pd.to_numeric(df_no_cat[column], errors='coerce')
    z_score = stats.zscore(df_no_cat[column])
    outliers = np.logical_or(z_score > 3, z_score < -3)
    outlier_count = np.sum(outliers)
    outlier_counts[column] = outlier_count

for column, count in outlier_counts.items():
    print(f"Column '{column}' has {count} outliers.")
 
#Check how many have missing values in rest of columns
data.isna().sum()

#Try imputing using kNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
knn_df = pd.DataFrame(imputer.fit_transform(df_no_cat),columns = df_no_cat.columns)

data.head(3)
knn_df.head(3)

#Split into training/test
df = knn_df
features = df.drop(columns=['drafted'])
features.info()
target = df['drafted']

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size = 0.2, random_state=22)
len(df)
len(feature_train)
len(feature_test)
len(target_train)
len(target_test)

#Check histograms
import matplotlib.pyplot as plt

knn_df.hist(bins=20, density=True, figsize=(8, 8))
plt.suptitle("Histograms of Columns")
plt.show()

from imblearn.over_sampling import SMOTE

#Balance dataset
smote = SMOTE(random_state=42)
feature_train_smote, target_train_smote = smote.fit_resample(feature_train, target_train)

len(feature_train_smote)
len(target_train_smote)
target_train_smote.value_counts()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Decision tree
decisiontree = DecisionTreeClassifier()
decisiontree.fit(feature_train_smote,target_train_smote)
dec_smote_pred = decisiontree.predict(feature_test)

#Calculate eval 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(target_test, dec_smote_pred)

precision = precision_score(target_test, dec_smote_pred)

recall = recall_score(target_test, dec_smote_pred)

f1 = f1_score(target_test, dec_smote_pred)

roc_auc = roc_auc_score(target_test, dec_smote_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)

#Random forest
rf = RandomForestClassifier()
rf.fit(feature_train_smote, target_train_smote)
rf_smote_pred = rf.predict(feature_test)
#Calculte eval
accuracy = accuracy_score(target_test, rf_smote_pred)

precision = precision_score(target_test, rf_smote_pred)

recall = recall_score(target_test, rf_smote_pred)

f1 = f1_score(target_test, rf_smote_pred)

roc_auc = roc_auc_score(target_test, rf_smote_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)

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
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(feature_train_scaled, target_train_smote)
logistic_pred_smote = logistic_model.predict(feature_test_scaled)
#Calculate eval
accuracy = accuracy_score(target_test, logistic_pred_smote)

precision = precision_score(target_test, logistic_pred_smote)

recall = recall_score(target_test, logistic_pred_smote)

f1 = f1_score(target_test, logistic_pred_smote)

roc_auc = roc_auc_score(target_test, logistic_pred_smote)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)

#SVM
from sklearn.svm import SVC
svm_model = SVC(kernel = 'linear')
svm_model.fit(feature_train_scaled, target_train_smote)
svm_pred_smote = svm_model.predict(feature_test_scaled)
#Calculate eval
accuracy = accuracy_score(target_test, svm_pred_smote)

precision = precision_score(target_test, svm_pred_smote)

recall = recall_score(target_test, svm_pred_smote)

f1 = f1_score(target_test, svm_pred_smote)

roc_auc = roc_auc_score(target_test, svm_pred_smote)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model_smote = KNeighborsClassifier(n_neighbors=3, metric='euclidean')  
knn_model_smote.fit(feature_train_scaled,target_train_smote)
array = feature_test_scaled.values
knn_pred_smote = knn_model_smote.predict(array)
#Calculate eval
accuracy = accuracy_score(target_test, dec_smote_pred)

precision = precision_score(target_test, dec_smote_pred)

recall = recall_score(target_test, dec_smote_pred)

f1 = f1_score(target_test, dec_smote_pred)

roc_auc = roc_auc_score(target_test, dec_smote_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)

#XGboost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3) 
xgb_model.fit(feature_train_scaled, target_train_smote)
xgb_pred = xgb_model.predict(feature_test_scaled)
#Calculate eval
accuracy = accuracy_score(target_test, xgb_pred)

precision = precision_score(target_test, xgb_pred)

recall = recall_score(target_test, xgb_pred)

f1 = f1_score(target_test, xgb_pred)

roc_auc = roc_auc_score(target_test, xgb_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC:", roc_auc)



