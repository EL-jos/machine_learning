import pandas as pd
import numpy as np

path = "test/adult.data"
cols = [
    "age", "workclass", "fnlwgt", "education", "education_num", 
    "marital_status", "occupation", "relationship", "race", "sex", 
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

df = pd.read_csv(path, header=None)
df.columns = cols

df = df.drop('education', axis=1)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

from sklearn.impute import SimpleImputer

si = SimpleImputer(missing_values=' ?', strategy='most_frequent')
X_train[:] = si.fit_transform(X_train)
X_test[:] = si.transform(X_test)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

nominal_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X_train_nominal = pd.DataFrame(
    ohe.fit_transform(X_train[nominal_cols]),
    columns=ohe.get_feature_names_out(nominal_cols),
    index=X_train.index
)
X_test_nominal = pd.DataFrame(
    ohe.transform(X_test[nominal_cols]),
    columns=ohe.get_feature_names_out(nominal_cols),
    index=X_test.index
)

X_train_encoded = pd.concat([X_train.drop(nominal_cols, axis=1), X_train_nominal], axis=1)
X_test_encoded = pd.concat([X_test.drop(nominal_cols, axis=1), X_test_nominal], axis=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

minmax = StandardScaler()
X_train_minmax = minmax.fit_transform(X_train_encoded)
X_test_minmax = minmax.transform(X_test_encoded)

cov_matrix = np.cov(X_train_minmax.T)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt

""" plt.bar(range(1, len(var_exp)+1), var_exp, align='center',
        label='Individual explained variance')
plt.step(range(1, len(cum_var_exp)+1), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('figures/05_02.png', dpi=300)
plt.show() """

eigen_pairs = [ (eigen_vals[i] , eigen_vecs[:, i]) for i in range(len(eigen_vals)) ]

eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack(
    (eigen_pairs[0][1].reshape(-1, 1), eigen_pairs[1][1].reshape(-1, 1))
)

X_train_pca = X_train_minmax.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=f'Class {l}', marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('figures/05_03.png', dpi=300)
plt.show()