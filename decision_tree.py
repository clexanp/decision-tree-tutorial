import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Prepare the data data
iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

X = data.drop('target', axis=1)
y = data[['target']]

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,
                                                y,
                                                test_size = 0.3,
                                                random_state = 42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred_dt = model.predict(X_test)

print('Akurasi',accuracy_score(y_test, y_pred_dt))