#I'm going to try something with this tesla dataset
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style

tesla_data = pd.read_csv('TSLA.csv')

X = np.array(tesla_data.drop(columns=['Date']))
y = np.array(tesla_data['Open'])
data_set = np.array(X[0])
print(X[0])
for i in range(2,len(X)+1):
    temp = np.mean(X[0:i], axis=0)
    data_set = np.vstack([data_set,temp])

result_set = y[1:]

X = data_set[0:2767]
Y = result_set[0:2767]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
predictions = linear.predict(X_test)
for X in range(len(predictions)):
    print(predictions[X], y_test[X])

acc =linear.score(X_test, y_test)
print(acc)
