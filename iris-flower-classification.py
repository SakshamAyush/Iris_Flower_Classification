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
dataset = pd.read_csv(url, names=names)
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