import pandas as pd
from scipy import stats
import numpy as np

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

#Try imputing mean values
df_mean = pd.DataFrame(data)
df_mean = df_mean.drop(columns=['team','conf','yr'])
df_mean.info()
for column in df_mean.columns:
    column_mean = df_mean[column].mean()
    df_mean[column].fillna(column_mean, inplace=True)
   
df_mean.isna().sum()

#Split into test/training set
#Split into training and test set
df = df_mean
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
