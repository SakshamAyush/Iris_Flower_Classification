# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 16:31:48 2018

@author: Saksham
"""
# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv("F:\College\Projects\Iris_Flower_Classification\data\iris.csv", names=names)
#to get total attributes
print(dataset.shape)
#avoid truncation
pd.options.display.max_columns = 10
#view first 5 data in dataset
print(dataset.head(5))
print("\n",dataset.describe())
#grouping data
print("\n",dataset.groupby('class').size())
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
#histogram
dataset.hist()
plt.show()
#multivariate plots
# scatter plot matrix
scatter_matrix(dataset)
plt.show()
array=dataset.values
X=array[:,0:4]
Y = array[:,4]
#X stores first 4 columns
#Y stores last column
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.25, random_state=0)
#we use accuracy here
scoring = 'accuracy'
#Checking Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=0)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# training with SVM as it has highest accuracy
sv = SVC()
sv.fit(X_train,Y_train)
predict =sv.predict(X_validation)
print(accuracy_score(Y_validation, predict))
print(confusion_matrix(Y_validation, predict))
print(classification_report(Y_validation, predict))