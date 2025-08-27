import pandas as pd
import numpy as np

path = "chp4/adult.data"
cols = [
    "age", "workclass", "fnlwgt", "education", "education_num", 
    "marital_status", "occupation", "relationship", "race", "sex", 
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]
df = pd.read_csv(path, header=None)

df.columns = cols

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

from sklearn.impute import SimpleImputer

si = SimpleImputer(missing_values=' ?', strategy='most_frequent')
X_train[:] = si.fit_transform(X_train)
X_test[:] = si.transform(X_test)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

X_train = X_train.drop('education', axis=1)
X_test = X_test.drop('education', axis=1)

nominal_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

ohe = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')

X_train_encoded = pd.DataFrame(
    ohe.fit_transform(X_train[nominal_cols]),
    columns=ohe.get_feature_names_out(nominal_cols),
    index=X_train.index
)
X_test_encoded = pd.DataFrame(
    ohe.transform(X_test[nominal_cols]),
    columns=ohe.get_feature_names_out(nominal_cols),
    index=X_test.index
)

X_train_nominal = pd.concat([X_train.drop(nominal_cols, axis=1), X_train_encoded], axis=1)
X_test_nominal = pd.concat([X_test.drop(nominal_cols, axis=1), X_test_encoded], axis=1)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train_nominal)
X_test_minmax = minmax.transform(X_test_nominal)

from sklearn.decomposition import PCA

# Réduction de dimension à 2D pour visualisation
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_minmax)
X_test_pca = pca.transform(X_test_minmax)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=10., random_state=1, solver='saga', max_iter=5000)
lr.fit(X_train_minmax, y_train)
score = lr.score(X_test_minmax, y_test)
print("Accuracy (test):", score)