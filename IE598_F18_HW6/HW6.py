
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Training and test set data were split into 90% and 10% respectively
iris = datasets.load_iris()
X = iris.data
y = iris.target
trainAccuracy = []
testAccuracy = []

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=i)
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    trainAccuracy.append(accuracy_score(y_train,y_train_pred))
    testAccuracy.append(accuracy_score(y_test,y_test_pred))
    inAccuracy = accuracy_score(y_train,y_train_pred)
    outAccuracy = accuracy_score(y_test,y_test_pred)
    print('\n Random state of', i, 'has an in-sample accuracy of', inAccuracy, 'and an out-of-sample accuracy', outAccuracy)
    
print('The in-sample mean for random test train split is', np.mean(trainAccuracy))
print('The in-sample standard deviation for random test train split is', np.std(trainAccuracy))
print('The out-of-sample mean for random test train split is', np.mean(testAccuracy))
print('The out-of-sample standard deviation for random test train split is', np.std(testAccuracy))

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
cvAccuracyIn = cross_val_score(estimator = tree, X = X_train, y = y_train, cv=10)
print('The cross-validation scores for in-sample are', cvAccuracyIn)
print('The cross-validation mean for in-sample is', np.mean(cvAccuracyIn))
print('The cross-validation standard deviation for in-sample is' ,np.std(cvAccuracyIn))

cvAccuracyOut = cross_val_score(tree, X, y, cv=10)
print('The cross-validation scores for out-of-sample are', cvAccuracyOut)
print('The cross-validation mean for out-of-sample is', np.mean(cvAccuracyOut))
print('The cross-validation standard deviation for out-of-sample is' ,np.std(cvAccuracyOut))

print("My name is Habeeb Olawin")
print("My NetID is: holawin2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


