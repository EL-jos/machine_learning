from urllib.error import HTTPError
import pandas as pd
import numpy as np

try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    print('From URL:', url)
    df = pd.read_csv(url, header=None, encoding='utf-8')
except HTTPError:
    url = 'wine.data'
    print('From local Iris path:', url)
    df = pd.read_csv(url, header=None, encoding='utf-8')

X = df.iloc[:, 1:]
y = df.iloc[:, 0]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

print(eigen_vals)