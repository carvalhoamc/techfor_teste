from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./input/dataset.csv')
df1 = df[['FEATURE 6', 'FEATURE 5', 'FEATURE 4', 'FEATURE 1', 'FEATURE 2',
       'FEATURE 3', 'Previsao A']]
df2 = df[['FEATURE 6', 'FEATURE 5', 'FEATURE 4', 'FEATURE 1', 'FEATURE 2',
       'FEATURE 3', 'Previsao B']]
df1.to_csv('./input/dataset1.csv',index=False)
df2.to_csv('./input/dataset2.csv',index=False)

print(Counter(df1['Previsao A']))
print(Counter(df2['Previsao B']))

Y = np.array(df1.iloc[:, -1])
X = np.array(df1.iloc[:, 0:-1])
ros = RandomOverSampler()
X_res, y_res = ros.fit_resample(X, Y)
y_res = y_res.reshape(len(y_res), 1)
df1 = pd.DataFrame(np.concatenate((X_res, y_res), axis=1))
df1.to_csv('./input/DF1.csv',index=False)
df1.columns = ['FEATURE 6', 'FEATURE 5', 'FEATURE 4', 'FEATURE 1', 'FEATURE 2',
       'FEATURE 3', 'Previsao A']

Y = np.array(df2.iloc[:, -1])
X = np.array(df2.iloc[:, 0:-1])
ros = RandomOverSampler()
X_res, y_res = ros.fit_resample(X, Y)
y_res = y_res.reshape(len(y_res), 1)
df2 = pd.DataFrame(np.concatenate((X_res, y_res), axis=1))
df2.to_csv('./input/DF2.csv',index=False)
df2.columns = ['FEATURE 6', 'FEATURE 5', 'FEATURE 4', 'FEATURE 1', 'FEATURE 2',
       'FEATURE 3', 'Previsao B']

print(Counter(df1['Previsao A']))
print(Counter(df2['Previsao B']))