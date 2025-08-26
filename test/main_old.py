import pandas as pd
import numpy as np

from io import StringIO
csv_data = """
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
"""

df1 = pd.read_csv(StringIO(csv_data))
from sklearn.impute import SimpleImputer
simpleImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df1_imputed = simpleImputer.fit_transform(df1)
new_df1 = pd.DataFrame(df1_imputed, columns=df1.keys())
#print(new_df1)

df2 = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])
df2.columns = ['color', 'size', 'price', 'classlabel']
size_mapping = {'M': 1, 'L': 2, 'XL': 3}
df2['size'] = df2['size'].map(size_mapping)
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
df2['classlabel'] = labelEncoder.fit_transform(df2['classlabel'])
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
color_encoder = ohe.fit_transform(df2['color'].to_numpy().reshape(-1, 1)).toarray()
df_color_encoder = pd.DataFrame(color_encoder, columns=ohe.get_feature_names_out(['color']))
new_df2 = pd.concat([df2.drop('color', axis=1), df_color_encoder], axis=1)
print(new_df2)


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None, encoding='utf-8')

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
X = df_wine.iloc[:, 1:].to_numpy()
y = df_wine.iloc[:, 0].to_numpy()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#print(X_train_std)