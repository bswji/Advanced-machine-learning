import pandas as pd

pd.set_option("display.max_columns" , None)

#Read data
data = pd.read_csv('Data/Week 1/train.csv')

data.info()

#Remove player_id, num, type
data['type'].value_counts()
data.drop(columns = ['type','player_id','num'], inplace=True)
data.info()

#Check columns with missing values
missing = data.isna().sum()
missing.to_csv('missing')

#Rec rank has 39055 missing values, dunk ratio has 30793, pick 54705 -Way too many missing values. Decided to remove.
data.drop(columns = 'pick', inplace = True)
data.drop(columns = 'Rec_Rank', inplace = True)
data.drop(columns = 'dunks_ratio', inplace = True)
data.info()

#Check how many have missing values in rest of columns
data.isna().sum()
len(data)