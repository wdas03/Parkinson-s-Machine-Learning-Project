# William Das
# Machine Learning Project

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, iqr

# Extract user info from file
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

# Read user data and insert into user dataset
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
columns_to_drop = ['BirthYear', 'DiagnosisYear', 'UPDRS', 'DA', 'MAOB', 'Other', 'Gender']
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
print(sufficient_typing_data_df)
print(sufficient_typing_data_df['Parkinsons'].value_counts())

# Create new df of typing data including users in the dataset with sufficient data
new_typing_data_df = typing_data_for_all[typing_data_for_all['user_id'].isin(sufficient_typing_data_df['user_id'].values)]

# Convert columns to all float values
new_typing_data_df.loc[:, 'Hold time'] = new_typing_data_df['Hold time'].apply(pd.to_numeric, downcast='float', errors='coerce')
new_typing_data_df.loc[:, 'Latency time'] = new_typing_data_df['Latency time'].apply(pd.to_numeric, downcast='float', errors='coerce')
new_typing_data_df.loc[:, 'Flight time'] = new_typing_data_df['Flight time'].apply(pd.to_numeric, downcast='float', errors='coerce')

# Filter out unwanted typing data
valid_keys = new_typing_data_df[(new_typing_data_df['Hold time'] > 0) & (new_typing_data_df['Latency time'] > 0) & (new_typing_data_df['Hold time'] < 3000) & (new_typing_data_df['Latency time'] < 3000)]
#valid_keys = new_typing_data_df
#valid_keys['Hold time'].fillna(new_typing_data_df['Hold time'].mean(), inplace=True)
#valid_keys['Latency time'].fillna(valid_keys['Latency time'].mean(), inplace=True)
#valid_keys['Flight time'].fillna(valid_keys['Flight time'].mean(), inplace=True)

valid_keys[valid_keys['Hold time'] < 0] = valid_keys['Hold time'].mean()

hold_time_stats = valid_keys[valid_keys.Hand.isin(['L', 'R'])].groupby(['user_id', 'Hand'])['Hold time'].agg([np.mean, skew, kurtosis, np.std, np.var, np.max, np.min, iqr, np.median])
latency_time_stats = valid_keys[valid_keys.Direction.isin(['LL', 'LR', 'RR', 'RL'])].groupby(['user_id', 'Direction'])['Latency time'].agg([np.mean, skew, kurtosis, np.std, np.var, np.max, np.min, iqr, np.median])

# Unstack and disperse values into columns for hold and latency times
hold_time_stats = hold_time_stats.unstack()
hold_time_stats.columns = ['_'.join(col).strip() + '_hold_time' for col in hold_time_stats.columns.values]
hold_time_stats['R_L_mean_difference'] = abs(hold_time_stats['mean_L_hold_time'] - hold_time_stats['mean_R_hold_time'])

latency_time_stats = latency_time_stats.unstack()
latency_time_stats.columns = ['_'.join(col).strip() + '_latency_time' for col in latency_time_stats.columns.values]
latency_time_stats['LL_RR_mean_difference'] = abs(latency_time_stats['mean_LL_latency_time'] - latency_time_stats['mean_RR_latency_time'])
latency_time_stats['LR_RL_mean_difference'] = abs(latency_time_stats['mean_LR_latency_time'] - latency_time_stats['mean_RL_latency_time'])

# Combine those statistics into one df
combined_stats = hold_time_stats.join(latency_time_stats)
#combined_stats['R_L_mean_difference'] = abs(combined_stats['mean_R_hold_time'] - combined_stats['mean_L_hold_time'])
#combined_stats['LR_RL_mean_difference'] = abs(combined_stats['mean_LR_latency_time'] - combined_stats['mean_RL_latency_time'])

combined_stats = combined_stats.reset_index()

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
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier

from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split, KFold, cross_val_score

features = [1] + list(range(3,63))
X = final_user_dataset.iloc[:, features].values
y = final_user_dataset.iloc[:, 0].values

# Hold times datasets
hold_features = [1] + list(range(6,15))
X_hold_times = hold_users_df.iloc[:, hold_features].values
y_hold_times = hold_users_df.iloc[:, 0].values

# Latency times datasets
latency_features = [1] + list(range(3, 24))
X_latency_times = latency_users_df.iloc[:, latency_features].values
y_latency_times = latency_users_df.iloc[:, 0].values


# Define classifiers
lr = LogisticRegression(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=15, random_state=0)
rf = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=0)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
nb = GaussianNB()
tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
svc = SVC(kernel='linear', random_state=0, probability=True)
mlp = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=0,
                    learning_rate_init=.1)
qda = QDA()
ada = AdaBoostClassifier(DecisionTreeClassifier(random_state=0), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)
eclf = VotingClassifier(estimators=[('lr', lr), ('gb', clf), ('rf', rf), ('knn', knn), ('nb', nb), ('tree', tree), ('svc', svc), ('qda', qda)], voting='soft')

#rfe = RFE(clf, 15)
#fit = rfe.fit(X, y)

#indices = [idx for idx in range(len(fit.support_)) if fit.support_[idx] == True]

#print(X.iloc[:, indices].columns)
#print(fit.ranking_)
#X = X.iloc[:, indices].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/5, random_state=0, stratify=y)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train, y_train)
X_test = sc_X.transform(X_test)

#X_train = lda.fit_transform(X_train, y_train)
#X_test = lda.transform(X_test)

# Scale independent variable features
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)



models = []
models.append(('Logistic Regression', lr))
models.append(('KNN', knn))
models.append(('Decision Tree', tree))
models.append(('Random Forest', rf))
models.append(('Naive Bayes', nb))
#models.append(('SVM', svc))
#models.append(('MLP', mlp))
models.append(('ADA', ada))
models.append(('Gradient', clf))
#models.append(('QDA', qda))
#models.append(('Voting', eclf))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    names.append(name)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    kfold = KFold(n_splits=10, random_state=0)
    kfold2 = KFold(n_splits=10, random_state=0)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    cv_results2 = cross_val_score(model, X_latency_times, y_latency_times, cv=kfold2, scoring=scoring)
    results.append(cv_results)
    msg2 = "K fold: %s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    msg3 = "(Latency) K fold %s: %f (%f)" % (name, cv_results2.mean(), cv_results2.std())
    msg = "%s: %f" % (name, accuracy)
    print(msg)
    print(msg2)
    print(msg3)



"""
# Data preprocessing
from sklearn.preprocessing import StandardScaler

features = list(range(3,53))
X = final_user_dataset.iloc[:, [1] + features]
y = final_user_dataset.iloc[:, 0].values

classifier = LogisticRegression(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=0)
rf = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=0)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
nb = GaussianNB()
tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
svc = SVC(kernel='linear', random_state=0)

# Feature selection
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

rfe = RFE(clf, 25)
fit = rfe.fit(X, y)

indices = [idx for idx in range(len(fit.support_)) if fit.support_[idx] == True]

print(X.iloc[:, indices].columns)
print(fit.ranking_)
X = X.iloc[:, indices].values

# Create train and test sets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/5, random_state=0, stratify=y)

# Scale independent variable features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import more libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

clf.fit(X_train, y_train)
print('Gradient:')
print(clf.score(X_test, y_test))

y_pred_boost = clf.predict(X_test)

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

# SVC
svc.fit(X_train, y_train)

print('SVC: ')
print(svc.score(X_test, y_test))

y_pred_svc = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_nb = confusion_matrix(y_test, y_pred_nb)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_svc = confusion_matrix(y_test, y_pred_svc)

print('Logistic Regression')
print(cm)

print('KNN')
print(cm_knn)

print('Naive Bayes')
print(cm_nb)

print('Decision Tree')
print(cm_dt)

print('Random Forest')
print(cm_rf)

print('SVC')
print(cm_svc)

"""