import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
#3from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



data = pd.read_csv("f.csv")
data.head(10)
data=data.replace('yes', 1)
data=data.replace('no', 0)
#data=data.replace('GOOD', 80)
data = data.dropna()
feature_df = data[['Engine load','rpm','throttle position','steering angledegree', 'break apply', 'gear position', 'vehicle speed']]
X = np.asarray(feature_df)
X[0:7]
y = np.asarray(data['accident'])
y [0:48]
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
yhat [0:5]

from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, clf.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
print(X_test.shape)
print(X_test)
Engineload = input('Engine Load : ')
rpm = input('rpm :')
throttle_position = input('throttle position :')
steeringangledegree = input('steeringangle in degree :')
break_apply = input('break apply :')
gear_position =input('gear position :')
vehicle_speed = input('vehicle speed :')
input_predict =clf.predict([[Engineload,rpm,throttle_position,steeringangledegree,break_apply,gear_position,vehicle_speed]]) # here  we gave the input

#print('prediction',clf.predict([[28.23529434,1507.75,16.47058868,90,1,2,26]]))
print('prediction',input_predict)

print(input_predict[0])

if input_predict[0] == 1:
    print("driver induced accident")
else:
    print("not driver induced accident")