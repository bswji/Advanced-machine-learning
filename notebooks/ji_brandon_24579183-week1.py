import pandas as pd

#Read data
data = pd.read_csv("https://raw.githubusercontent.com/bswji/Advanced-machine-learning/main/Data/Week%201/train.csv?token=GHSAT0AAAAAACGNNF36GLG4QQMGBEX2ALZIZG6ESAQ")

data = pd.DataFrame(data)
data.info()

#Clean data
#Remove player id
data = data.drop(columns = "player_id")
data.info()
#Check for missing values
data.isnull().sum()
len(data)
