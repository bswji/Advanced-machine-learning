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
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(feature_train_scaled, target_train_smote)
logistic_pred_smote = logistic_model.predict(feature_test_scaled)
roc_log_smote = roc_auc_score(target_test, logistic_pred_smote)
print(roc_log_smote)

#SVM
from sklearn.svm import SVC
svm_model = SVC(kernel = 'linear')
svm_model.fit(feature_train_scaled, target_train_smote)
svm_pred_smote = svm_model.predict(feature_test_scaled)
roc_svm_smote = roc_auc_score(target_test, svm_pred_smote)
print(roc_svm_smote)
