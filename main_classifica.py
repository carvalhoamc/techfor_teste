import warnings
from collections import Counter
warnings.filterwarnings('ignore')

import joblib

import category_encoders as ce
from imblearn.metrics import classification_report_imbalanced
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

datasets = ['dataset1','dataset2']

dict_categorical_cols = {'dataset1': [0, 1, 2, 3, 4],
                         'dataset2': [0, 1, 2, 3, 4],
                         }

ia_ml_model_dataset1 = joblib.load('./componentes/model_ia_dataset1_3.ml.gzip')
ia_ml_model_dataset2 = joblib.load('./componentes/model_ia_dataset2_3.ml.gzip')

for filename in datasets:
	fname = './input/' + filename + '.csv'
	print(fname)
	df_data = pd.read_csv(fname, header=None)
	drop_na_col = False  ## auto drop columns with nan's (bool)
	drop_na_row = True  ## auto drop rows with nan's (bool)
	drop_null_col = True
	drop_null_row = True
	
	if df_data.isnull().values.any():
		raise ValueError("cannot proceed: data cannot contain NaN values")
	
	Y = np.array(df_data.iloc[:, -1])
	X = np.array(df_data.iloc[:, 0:-1])
	le = LabelEncoder()
	Y = le.fit_transform(Y)
	print(Counter(Y))
	
	
	







