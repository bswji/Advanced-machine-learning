import pandas as pd
pd.set_option("display.max_columns" , None)

#Read data
data = pd.read_csv("https://raw.githubusercontent.com/bswji/Advanced-machine-learning/main/Data/Week%201/train.csv?token=GHSAT0AAAAAACGH327TWKJY4MOW7BSYEQW2ZHAUQEA")
data.info()

#Drop irrelevant columns
data = data.drop(columns = ["player_id", "num", "type"])


#Check missing values
data.isnull().sum()
<<<<<<< Updated upstream

#Drop columns that contain >100 missing values.
data_wo_na = data.drop(columns = ["rimmade", "rimmade_rimmiss", "midmade", "midmade_midmiss", "mid_ratio", "dunksmade","dunksmiss_dunksmade","dunks_ratio","pick", "Rec_Rank", "ast_tov", "rim_ratio", "yr"])
data_wo_na.isna().sum()

#Drop all rows that contain missing values
len(data_wo_na)
data_no_na = data_wo_na.dropna()
data_no_na.isna().sum()
len(data_no_na)

#Check categorical values (team, conf, ht, all)
data_no_na['team'].nunique()
data_no_na['conf'].nunique()
data_no_na['ht'].nunique()
data_no_na['type'].nunique()

#Remove categorical values
data_removed = data_no_na.drop(columns = ["team", "conf", "ht"])

### Modelling ###

from sklearn.tree import DecisionTreeClassifier #Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#Split into training and test set
df = data_removed
features = df.drop(columns=['drafted'])
features.info()
target = df['drafted']
target.info()

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

#Calculate ROC
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(target_test, ypred)
print(roc_auc)

#Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rfclass = rf.fit(feature_train, target_train)
y_pred = rfclass.predict(feature_test)
rf_roc_auc = roc_auc_score(target_test, y_pred)
print(rf_roc_auc)

#Tune hyper parameters
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20)}
rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)


rand_search.fit(feature_train, target_train)

best_rf = rand_search.best_estimator_
print(best_rf)

rf = RandomForestClassifier(n_estimators= 207, max_depth = 11)
optimisedrf = rf.fit(feature_train, target_train)
optimised_pred = optimisedrf.predict(feature_test)
optimisedroc = roc_auc_score(target_test, optimised_pred)
print(optimisedroc)

=======
len(data)
po
>>>>>>> Stashed changes
