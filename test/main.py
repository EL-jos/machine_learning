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

simpleImputer = SimpleImputer(missing_values=" ?", strategy="most_frequent")

X_train_df = pd.DataFrame(X_train, columns=cols[:-1])
X_test_df = pd.DataFrame(X_test, columns=cols[:-1])

X_train_df[:] = simpleImputer.fit_transform(X_train_df)
X_test_df[:] = simpleImputer.transform(X_test_df)

from sklearn.preprocessing import OneHotEncoder, StandardScaler

nominal_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']


for nominal_col in nominal_cols:
    ohe = OneHotEncoder(drop=None)

    train_encoded = ohe.fit_transform(X_train_df[nominal_col].to_numpy().reshape(-1, 1)).toarray()
    test_encoded = ohe.fit_transform(X_test_df[nominal_col].to_numpy().reshape(-1, 1)).toarray()

    train_encoded_df = pd.DataFrame(train_encoded, columns=ohe.get_feature_names_out([nominal_col]))
    test_encoded_df = pd.DataFrame(test_encoded, columns=ohe.get_feature_names_out([nominal_col]))

    X_train_encoded = pd.concat([X_train_df.drop(nominal_col, axis=1), train_encoded_df], axis=1)
    X_test_encoded = pd.concat([X_test_df.drop(nominal_col, axis=1), test_encoded_df], axis=1)

sc = StandardScaler()
X_train_encoded_std = sc.fit_transform(X_train_encoded)
X_test_encoded_std = sc.transform(X_test_encoded)
