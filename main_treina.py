import warnings
from collections import Counter

from imblearn.over_sampling import RandomOverSampler, SMOTE

warnings.filterwarnings('ignore')

import joblib

import category_encoders as ce
from gsmote import GeometricSMOTE
from imblearn.metrics import classification_report_imbalanced
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

datasets = ['dataset2']

dict_categorical_cols = {#'dataset1': [0, 1, 2, 3, 4],
                         'dataset2': [0, 1, 2, 3, 4],
                         }
dfcol = ['ID', 'DATASET', 'FOLD', 'PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA']
df = pd.DataFrame(columns=dfcol)
i = 0
for filename in datasets:
	fname = './input/' + filename + '.csv'
	print(fname)
	df_data = pd.read_csv(fname)
	df_data = df_data.drop_duplicates()
	drop_na_col = False  ## auto drop columns with nan's (bool)
	drop_na_row = True  ## auto drop rows with nan's (bool)
	drop_null_col = True
	drop_null_row = True
	
	if bool(drop_na_col) == True:
		df_data = df_data.dropna(axis=1)  ## drop columns with nan's
	
	if bool(drop_na_row) == True:
		df_data = df_data.dropna(axis=0)  ## drop rows with nan's
	
	if df_data.isnull().values.any():
		raise ValueError("cannot proceed: data cannot contain NaN values")
		
	Y = np.array(df_data.iloc[:, -1])
	X = np.array(df_data.iloc[:, 0:-1])
	le = LabelEncoder()
	Y = le.fit_transform(Y)
	joblib.dump(le, './componentes/label_' + filename +'.gzip')
	print(Counter(Y))
	target = ce.TargetEncoder(cols=dict_categorical_cols[filename])
	X= target.fit_transform(X, Y)
	joblib.dump(target, './componentes/encoder_' + filename +  '.gzip')
	ros = RandomOverSampler()
	X, Y = ros.fit_resample(X, Y)
	X = X.to_numpy()
	
	skf = StratifiedKFold(n_splits=50, shuffle=True)
	fold = 0
	for train_index, test_index in skf.split(X, Y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		y_train = y_train.reshape(len(y_train), 1)
		y_test = y_test.reshape(len(y_test), 1)
		print('Folder = ', fold)
		clf = KNeighborsClassifier()
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		joblib.dump(clf, './componentes/model_ia_' + filename +'_'+ str(fold) + '.ml.gzip')
		res = classification_report_imbalanced(y_test, y_pred,digits=4)
		print(res)
		identificador = filename
		aux = res.split()
		score = aux[-7:-1]
		df.at[i, 'ID'] = identificador
		df.at[i, 'DATASET'] = filename
		df.at[i, 'FOLD'] = fold
		df.at[i, 'PRE'] = score[0]
		df.at[i, 'REC'] = score[1]
		df.at[i, 'SPE'] = score[2]
		df.at[i, 'F1'] = score[3]
		df.at[i, 'GEO'] = score[4]
		df.at[i, 'IBA'] = score[5]
		i = i + 1
		fold = fold + 1
	df.to_csv('./output/results.csv', index=False)













