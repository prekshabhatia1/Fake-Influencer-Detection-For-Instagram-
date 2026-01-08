import pandas as pd 
import numpy as np

df = pd.read_csv("C:\\Users\\DeLL\\Downloads\\dataset.csv") #Because Python sees backslashes \ as escape characters. ,gives unescape error
print(df.shape)

#CLEANING THE DATA 
df.columns=df.columns.str.strip().str.replace('\n','')       #remove unwanted whitespace

def convert_to_numeric_in_millions(value):
  if isinstance(value, str):
    if 'M' in value or 'm' in value:
        # Value is in Millions
        return float(value.replace('M', '').replace('m' , ''))
    elif 'K' in value or 'k' in value:
        # Value is in Thousands, convert to Millions by dividing by 1000
        return float(value.replace('K', '').replace('k',''))/ 1000

    else:
        # Handle values that are just numbers as strings (e.g., "0")
        try:
           return float(value) / 1000000    #to handle for eg 400 foloowers 
        except ValueError:
           return np.nan                    #If the code in the try block fails because the string cannot be converted into a number ValueError is raised).
  return np.nan
numeric_cols = ["followers", "avg_likes", "total_likes", "new_post_avg_like"]
for col in numeric_cols :
  df[col] = df[col].apply(convert_to_numeric_in_millions)
df.fillna(0,inplace = True)
  
print(df.columns)
print(df.isnull().sum())                          #identify missing values 

print(df.isnull().sum())  
df["60_day_eng_rate"] = (
    df["60_day_eng_rate"]
    .astype(str)
    .str.replace('%', '')
    .astype(float)  
    / 100
)
#replaces every missing values with 0 inplace means The original dfis updated directly
 #df.to_csv("C:\\Users\\DeLL\\Downloads\\dataset_cleaned.csv", index=False)    #index=false it will ot returm unwanted index we will use index=true whem the i ndex is meaningfiull
#CLEANING IS DONE

#FEATURE EXTRACTION 
df["like_follower_ratio"] = df["avg_likes"] / (df["followers"] + 1)
 #df.loc is used to select and update rows and columns by condition or label in a pandas DataFrame.
df["likes_per_post"] = df["total_likes"] / (df["avg_likes"] + 1)
df["engagement_per_follower"] = df["60_day_eng_rate"] / (df["followers"] + 1)

features = [
    "followers",
    "avg_likes",
    "new_post_avg_like",
    "like_follower_ratio",
    "engagement_per_follower"
]

X = df[features]

from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # assume ~5% fake influencers
    random_state=42
)

iso.fit(X)

df["anomaly_label"] = iso.predict(X)

df["anomaly_score"] = iso.decision_function(X)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso.fit(X_scaled)



import joblib
joblib.dump(iso, "isolation_forest_influencer.pkl")








