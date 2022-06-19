import copy
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def encoder(obj):
    obj = copy.deepcopy(obj)
    le = LabelEncoder()
    # transform binary feature
    for col in obj:
        if obj[col].dtype == 'object':
            if len(list(obj[col].unique())) <= 2:
                le.fit(obj[col])
                obj[col] = le.transform(obj[col])
    # one hot encode multivalued features
    obj = pd.get_dummies(obj)

    return obj