#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: williamdas
"""

# Import libraries
import matplotlib.pyplot as plt

# Import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

# Import metrics
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support, average_precision_score

# Import data preprocessing libraries
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.feature_selection import SelectKBest

lr = LogisticRegression(random_state=0)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=5, random_state=0)
rf = RandomForestClassifier(n_estimators=200)
knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
nb = GaussianNB()
tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
svc = SVC(kernel='linear', random_state=0)
ada = AdaBoostClassifier(DecisionTreeClassifier(random_state=0), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)
vote = VotingClassifier(estimators=[('dt', tree),('gb', gb), ('ada', ada)], voting='hard')

models = []

#models.append(('Logistic Regression', lr))
models.append(('KNN', knn))
models.append(('Decision Tree', tree))
models.append(('Random Forest', rf))
models.append(('Naive Bayes', nb))
#models.append(('SVM', svc))
#models.append(('MLP', mlp))
models.append(('Gradient', gb))
#models.append(('Voting', vote))
models.append(('Ada', ada))

# Hyperparameter optimization
def gridsearch():
    pass

# Empirical evaluation of all models: cross validation and test set accuray
def run_models(X, y, test_size=2/5, cv=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)
    for name, model in models:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        cross = cross_val_score(model, X_test, y_test, cv=cv)
        
        print(name, 'CV:', sum(cross) / len(cross))
        #print(cross)
        print(name, 'Accuracy:', accuracy)
        y_pred = model.predict(X_test)
        
        print('AUC:', get_auc(y_test, y_pred))
        print('Average precision:', average_precision_score(y_test, y_pred))
        plot_roc(y_test, y_pred)
# Select K best features based on statistical tests
def selectkbest(x, y, k=10):
    return SelectKBest(k=k).fit_transform(x, y)

# Iterate through all possible select k best instances
def selectkbest_loop(x, y, test_size=.3):
    for i in range(15, 1, -1):
        print("\nNumber of features:", i)
        new = selectkbest(x, y, i)
        run_models(x, y, test_size=test_size)

def get_precision_recall_f1(y_test, y_pred):
    return precision_recall_fscore_support(y_test, y_pred)

# Get AUC Values
def get_auc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

# Plot AUROC 
def plot_roc(y_test, y_pred, filename="roc.png"):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig(filename)

# Get Confusion Matrix 
def get_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)

