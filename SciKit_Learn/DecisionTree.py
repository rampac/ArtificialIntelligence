# A decision tree is one of most frequently and widely used supervised machine learning algorithms that can perform both regression and classification tasks.

#classification solution
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline

dataset = pd.read_csv("D:/Datasets/bill_authentication.csv")  

dataset.shape 

dataset.head()  

X = dataset.drop('Class', axis=1)  
y = dataset['Class']  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test) 

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

#Regression soultion

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline

dataset = pd.read_csv('D:\Datasets\petrol_consumption.csv')  

dataset.describe()

X = dataset.drop('Petrol_Consumption', axis=1)  
y = dataset['Petrol_Consumption'] 

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
df 

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

