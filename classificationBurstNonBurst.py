import numpy as np
import array as arr
from sklearn.linear_model import LogisticRegression as LR
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, explained_variance_score, r2_score
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
# visualize the data
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import cross_val_score
import time
from sklearn.neural_network import MLPClassifier

#reading the files
data_file = 'features.csv'
label_file = 'label.csv'
X = np.loadtxt(open(data_file, "rb"), delimiter=",", dtype=np.int16)
y = np.loadtxt(open(label_file, "rb"), delimiter=",", dtype=np.int16)

#dividing traning and test data
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.30, random_state=1, shuffle=True)

#For summarizing the models, all models are then discussed in detail later
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='saga', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM (linear)', SVC(kernel='linear')))
models.append(('SVM (poly)', SVC(gamma='auto')))
models.append(('ANN', MLPClassifier()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.40, random_state=1, shuffle=True)

#Detailed anaysis with classfication reports
#KNN
tic = time.perf_counter()
clf = KNeighborsClassifier(n_neighbors=5)
# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)
#Predict the response for training dataset
X_pred = clf.predict(X_train)
#Predict the response for test dataset
Y_pred = clf.predict(X_test)
print('For value of k= ' + str(5))
print('Accuracy of test data:')
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
#Printing Confusion matrix for the test data
#Classification Report for test data
#print(classification_report(Y_test, Y_pred))
#Stoping timer
toc = time.perf_counter()
print(f"Time taken to train the model {toc - tic:0.4f} seconds")


#SVM
#Staring the timer
tic = time.perf_counter()
print('For SVM with kernel function poly ')
clf = SVC(kernel='poly')
# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)
#Predict the response for training dataset
X_pred = clf.predict(X_train)
#Predict the response for test dataset
Y_pred = clf.predict(X_test)
print('Accuracy of test data:')
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
#Printing Confusion matrix for the test data
#Classification Report for test data
#print(classification_report(Y_test, Y_pred))
#Stoping timer
toc = time.perf_counter()
print(f"Time taken to train the model {toc - tic:0.4f} seconds")



#LR
model=LogisticRegression(solver='saga', multi_class='ovr')
print('For logistic regression with ')
model = model.fit(X_train,Y_train)
#Predict the response for training dataset
X_pred = model.predict(X_train)
#Predict the response for test dataset
Y_pred = model.predict(X_test)
print('Accuracy of test data:')
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
#Printing Confusion matrix for the test data
#Classification Report for test data
#print(classification_report(Y_test, Y_pred))
#Stoping timer
toc = time.perf_counter()
print(f"Time taken to train the model {toc - tic:0.4f} seconds")

#Decision Tree
model=DecisionTreeClassifier()
print('For Decision Tree ')
model = model.fit(X_train,Y_train)
#Predict the response for training dataset
X_pred = model.predict(X_train)
#Predict the response for test dataset
Y_pred = model.predict(X_test)
print('Accuracy of test data:')
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
#Printing Confusion matrix for the test data
#Classification Report for test data
#print(classification_report(Y_test, Y_pred))
#Stoping timer
toc = time.perf_counter()
print(f"Time taken to train the model {toc - tic:0.4f} seconds")

#Gaussian NB
model=GaussianNB()
print('For Gaussian NB')
model = model.fit(X_train,Y_train)
#Predict the response for training dataset
X_pred = model.predict(X_train)
#Predict the response for test dataset
Y_pred = model.predict(X_test)
print('Accuracy of test data:')
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
#Printing Confusion matrix for the test data
#Classification Report for test data
#print(classification_report(Y_test, Y_pred))
#Stoping timer
toc = time.perf_counter()
print(f"Time taken to train the model {toc - tic:0.4f} seconds")

#ANN
print('For ANN : ')
mlp = MLPClassifier()
t = time.time()
#Fitting the model
mlp.fit(X_train, Y_train)
t_mlp = time.time() - t
#Predicting Values
y_pred_mlp = mlp.predict(X_test)
y_probs=mlp.predict_proba(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred_mlp))
#print(classification_report(Y_test, y_pred_mlp))

