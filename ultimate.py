# William Das
# Early Diagnosis of Parkinson's Machine Learning Project

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, iqr

# Define functions for serializing variables using pickle libary
import pickle

def serialize_variable(var, filename):
    pickle.dump(var, open(os.path.join('stored_variables', filename), 'wb'), protocol=4)

def load_variable_from_file(filename):
    var = pickle.load(open(os.path.join('stored_variables', filename), 'rb'))
    return var
    
# Define function to extract user info from a given file
def extract_user_data(filename):
    file = open('users/' + filename, 'r')
    user_id = filename[5:15]
    data = [user_id]
    for line in file.readlines():
        data.append(line.split(": ")[-1].strip())
    return data

# Get list of files in each directory 
user_files = sorted(os.listdir("users"))
data_files = sorted(os.listdir("data"))
user_ids = [filename[:10] for filename in user_files]

# Create user dataset
user_columns = ['user_id', 'BirthYear', 'Gender', 'Parkinsons', 'Tremors', 'DiagnosisYear', 'Sided', 'UPDRS', 'Impact', 'Levadopa', 'DA', 'MAOB', 'Other']
user_dataset = pd.DataFrame(columns = user_columns)

# Read all user files and insert into user dataset
for idx in range(len(user_files)):
    filename = user_files[idx]
    user_id = filename[5:15]
    if user_id in [s[:10] for s in data_files]:
        user_dataset.loc[idx] = extract_user_data(filename)
        
# Encode binary variables
binary_vars = ['Parkinsons', 'Tremors', 'Levadopa']
for b in binary_vars:
    user_dataset[b] = user_dataset[b].replace({'True': 1, 'False': 0})
    
# Categorical dummy variables
categorical_vars = ['Sided']
user_dataset = pd.get_dummies(user_dataset, columns=categorical_vars)

# Clean user dataset up a bit
user_dataset.loc[~user_dataset.Impact.isin(['Mild', 'Severe', 'Medium']), 'Impact'] = np.nan

# Use patients only with mild severity if diagnosed and not taking levadopa
user_dataset = user_dataset[((user_dataset.Parkinsons == 1) & (user_dataset.Impact == 'Mild')) | ((user_dataset.Parkinsons == 0) & (user_dataset.Levadopa == 0))]

# Drop necessary columns
columns_to_drop = ['BirthYear', 'DiagnosisYear', 'UPDRS', 'DA', 'MAOB', 'Other', 'Gender', 'Levadopa']
user_dataset = user_dataset.drop(columns=columns_to_drop)

# Get all typing data
typing_data_columns = ['user_id', 'Date', 'Timestamp', 'Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time']
typing_data_for_all = pd.DataFrame(columns=typing_data_columns)

for idx in range(len(data_files)):
    filename = data_files[idx]
    user_id = filename[:10]
    # Process file if id is in out user dataset
    if user_id in user_dataset['user_id'].values:
        data = pd.read_table('data/' + filename, sep='\t', names=typing_data_columns, index_col=False, low_memory=False)
        typing_data_for_all = typing_data_for_all.append(data)
        
# Get typing data where value counts of user data is over 2000 keystrokes
sufficient_typing_data_boolean_df = typing_data_for_all['user_id'].value_counts() > 2000  # Boolean df of values
sufficient_values = sufficient_typing_data_boolean_df[sufficient_typing_data_boolean_df == True].index.values
sufficient_typing_data_df = user_dataset[user_dataset['user_id'].isin(sufficient_values)]
#print(sufficient_typing_data_df)
print(sufficient_typing_data_df['Parkinsons'].value_counts())

# Create new df of typing data including users in the dataset with sufficient data
typing_data_for_all = typing_data_for_all[typing_data_for_all['user_id'].isin(sufficient_typing_data_df['user_id'].values)]

# Convert columns to all float values
typing_data_for_all.loc[:, 'Hold time'] = typing_data_for_all['Hold time'].apply(pd.to_numeric, downcast='float', errors='coerce')
typing_data_for_all.loc[:, 'Latency time'] = typing_data_for_all['Latency time'].apply(pd.to_numeric, downcast='float', errors='coerce')
typing_data_for_all.loc[:, 'Flight time'] = typing_data_for_all['Flight time'].apply(pd.to_numeric, downcast='float', errors='coerce')

# Filter out unwanted typing data
valid_keys = typing_data_for_all[(typing_data_for_all['Hold time'] > 0) & (typing_data_for_all['Latency time'] > 0) & (typing_data_for_all['Hold time'] < 2000) & (typing_data_for_all['Latency time'] < 2000)]
hold_time_stats = valid_keys[valid_keys.Hand.isin(['L', 'R', 'S'])].groupby(['user_id', 'Hand'])['Hold time'].agg([np.mean, np.median, skew, kurtosis, np.max, np.min, iqr])
latency_time_stats = valid_keys[valid_keys.Direction.isin(['LL', 'LR', 'RR', 'RL'])].groupby(['user_id', 'Direction'])['Latency time'].agg([np.mean, np.median, skew, kurtosis, np.max, np.min, iqr])

# Unstack and disperse values into columns for hold and latency times
hold_time_stats = hold_time_stats.unstack()
hold_time_stats.columns = ['_'.join(col).strip() + '_hold_time' for col in hold_time_stats.columns.values]

latency_time_stats = latency_time_stats.unstack()
latency_time_stats.columns = ['_'.join(col).strip() + '_latency_time' for col in latency_time_stats.columns.values]

# Combine those statistics into one df
combined_stats = hold_time_stats.join(latency_time_stats)
combined_stats['user_id'] = combined_stats.index.values
print(len(combined_stats))

# Combine stats into user dataset
final_user_dataset = pd.merge(user_dataset, combined_stats, on='user_id')
final_user_dataset = final_user_dataset.drop(columns=['user_id', 'Impact']) # Drop user id column as it is no longer needed

latency_time_stats = latency_time_stats.reset_index()
hold_time_stats = hold_time_stats.reset_index()

latency_users_df = pd.merge(user_dataset, latency_time_stats, on='user_id').drop(columns=['user_id', 'Impact'])
hold_users_df = pd.merge(user_dataset, hold_time_stats, on='user_id').drop(columns=['user_id', 'Impact'])

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
from sklearn.metrics import classification_report, roc_auc_score

# Data preprocessing
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split, KFold, cross_val_score

X_latency = latency_users_df.iloc[:, 1:].values
y_latency = latency_users_df.iloc[:, 0].values

X = final_user_dataset.iloc[:, 1:].values
y = final_user_dataset.iloc[:, 0].values

sc = StandardScaler()
X_scaled = sc.fit_transform(X_latency)
X_normalized = normalize(X_latency)

lr = LogisticRegression(random_state=0)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=5, random_state=0)
rf = RandomForestClassifier(n_estimators=100)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
nb = GaussianNB()
tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
svc = SVC(kernel='linear', random_state=0)
ada = AdaBoostClassifier(DecisionTreeClassifier(random_state=0), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)
vote = VotingClassifier(estimators=[('dt', tree),('gb', gb), ('ada', ada)], voting='hard')

def run_models(X, y):
    models = []
    models.append(('Logistic Regression', lr))
    models.append(('KNN', knn))
    models.append(('Decision Tree', tree))
    models.append(('Random Forest', rf))
    models.append(('Naive Bayes', nb))
    #models.append(('SVM', svc))
    #models.append(('MLP', mlp))
    models.append(('Gradient', gb))
    #models.append(('QDA', qda))
    models.append(('Voting', vote))
    models.append(('Ada', ada))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=0)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        accuracy = cv_results.mean()
        msg = "K fold: %s: %f" % (name, accuracy)
        print(msg)

run_models(X_scaled, y_latency)

# Import more libraries
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print('Logistic Regression: ')
print(accuracy)

y_pred = classifier.predict(X_test)

# Random Forest Classifier
rf.fit(X_train, y_train)
accuracy_rf = rf.score(X_test, y_test)

print('Random Forest Classifier: ')
print(accuracy_rf)

y_pred_rf = rf.predict(X_test)

# KNN Neighbors
knn.fit(X_train, y_train)
accuracy_knn = knn.score(X_test, y_test)

print('KNN: ')
print(accuracy_knn)

y_pred_knn = knn.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

print('Naive Bayes: ')
print(nb.score(X_test, y_test))

y_pred_nb = nb.predict(X_test)

# Decision Tree Classifier
tree.fit(X_train, y_train)

print('Decision Tree: ')
print(tree.score(X_test, y_test))

y_pred_dt = tree.predict(X_test)

print(cross_val_score(tree, X, y, cv=KFold(10), scoring='accuracy').mean())

print(classification_report(y_test, y_pred_dt))
print(roc_auc_score(y_test, y_pred_dt))
# SVC
svc.fit(X_train, y_train)

print('SVC: ')
print(svc.score(X_test, y_test))

y_pred_svc = svc.predict(X_test)

ada.fit(X_train, y_train)

print('ADA: ')
print(ada.score(X_test, y_test))
y_pred_ada = ada.predict(X_test)

print(cross_val_score(ada, X, y, cv=KFold(10), scoring='accuracy').mean())

print(classification_report(y_test, y_pred_ada))

gb.fit(X_train, y_train)

print('GB:')
print(gb.score(X_test, y_test))

y_pred_gb = gb.predict(X_test)

print(classification_report(y_test, y_pred_gb))

vote.fit(X_train, y_train)

print('Vote:')
print(vote.score(X_test, y_test))

cm = confusion_matrix(y_test, y_pred)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_nb = confusion_matrix(y_test, y_pred_nb)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_svc = confusion_matrix(y_test, y_pred_svc)
cm_ada = confusion_matrix(y_test, y_pred_ada)
cm_gb = confusion_matrix(y_test, y_pred_gb)

print('Logistic Regression')
print(cm)

print('KNN')
print(cm_knn)

print('Naive Bayes')
print(cm_nb)

print('Decision Tree')
print(cm_dt)

print('ADA')
print(cm_ada)

print('GB')
print(cm_gb)

print('Random Forest')
print(cm_rf)

print('SVC')
print(cm_svc)

fpr, tpr, thresholds = roc_curve(y_pred, y_pred_dt)
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


