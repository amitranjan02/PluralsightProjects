import numpy as np 
import pandas as pd 
  
df = pd.read_csv('iris.csv') 
print("unique species:") 
print(df['Name'].unique())

# perform index encoding:
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'Name':
df['Name'] = label_encoder.fit_transform(df['Name']) 
  
print("index-encoded species:") 
print(df['Name'].unique())

