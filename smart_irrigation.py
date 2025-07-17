import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
import joblib
df = pd.read_csv("68544eadb00637051626.csv")
print("Dataset Preview")
print(df.head())
df.info()
df.columns
df = df.drop('Unnamed: 0',axis=1)
df.head()
df.describe()
x = df.iloc[:,0:20]
y =df.iloc[:,20]
x.sample(10)
y.sample(10)
x.info()
y.info()
x.shape,y.shape