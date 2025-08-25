import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

csv_data = """
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
"""
""" df = pd.read_csv(StringIO(csv_data))
print(df.dropna(subset=['C', 'D'])) """

""" df = pd.DataFrame([
['green', 'M', 10.1, 'class2'],
['green', 'L', 13.5, 'class1'],
['blue', 'XL', 15.3, 'class2']
])

df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {'M': 1, 'L': 2, 'XL': 3}
df['size'] = df['size'].map(size_mapping)

labelEncoder = LabelEncoder()
df['classlabel'] = labelEncoder.fit_transform(df['classlabel'])

X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_encoded = color_ohe.fit_transform(df['color'].values.reshape(-1, 1)).toarray()

df_color = pd.DataFrame(color_encoded, columns=color_ohe.get_feature_names_out(['color']))
df = pd.concat([df.drop('color', axis=1), df_color], axis=1)
print(df) """

df_wine = pd.read_csv('chp4/wine.data.txt', header=None, encoding='utf-8')


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

#print('Class labels', np.unique(df_wine['Class label']), df_wine.isnull().sum())

from sklearn.model_selection import train_test_split
X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#df_wine.head()
