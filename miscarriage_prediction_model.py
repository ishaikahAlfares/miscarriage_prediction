import pandas as pd
from sklearn.model_selection import cross_validate, KFold, cross_val_predict
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from xgboost.sklearn import XGBClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss


# import the dataset
data2 = pd.DataFrame(pd.read_csv('D:\shaik\Documents\Project\implementation\dataset.csv'))
print(data2.head())

x1 = data2[['ga_days', 'c08crl_av', 'mage', 'Age_cat', 'BMI', 'parity', 'Education']]  # create features array
y1 = data2[['preg_outcome']]  # create target array

print('--shape of original dataset \n',x1.shape, y1.shape)
count= pd.value_counts(data2['preg_outcome'], sort=True)
print('--count of original dataset\n',count)

#----------------Combined resampling-------------------------
smk= SMOTEENN(random_state=42)
x,y=smk.fit_resample(x1,y1)
mis = y[y['preg_outcome'] == 2]
preg = y[y['preg_outcome'] == 1]
print('--Count of resampeled dataset \n',mis.shape, preg.shape)

print('--shape of new dataset \n',x.shape, y.shape)

# choose a classifier
clf = RandomForestClassifier(random_state=42)
#clf = DecisionTreeClassifier(random_state=42)
#clf = KNeighborsClassifier()
#clf = LogisticRegression(random_state=42)
#clf = svm.SVC(kernel='linear')
#clf = XGBClassifier(random_state=42)


# create a KFold to choose number of folds and random state
cv = KFold(n_splits=10, random_state=42, shuffle=True)

# do a cross validation using cross_validate to obtain
# training and testing scores to calculate training and testing errors
cv_results = cross_validate(clf.fit(x, y.values.ravel()), x, y.values.ravel(), groups=None,scoring=None, cv=cv, n_jobs=-1,
                            verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=True)
# do a cross validation using cross_val_predict to be able to use it with
# confusion matrix to calculate the rest of evaluation measures
cv_results2 = cross_val_predict(clf.fit(x, y.values.ravel()), x, y.values.ravel(), groups=None, cv=cv, n_jobs=-1,
                                verbose=0, fit_params=None, pre_dispatch='2*n_jobs')

print('-' * 50)
print('Training Error: ', 1-cv_results['train_score'].mean())
print('Testing Error: ', 1-cv_results['test_score'].mean())
test_acc = accuracy_score(y, cv_results2)
mis_classification = 1-test_acc
Sensitivity = recall_score(y, cv_results2)
Precision = precision_score(y, cv_results2)
f1score = f1_score(y, cv_results2)
print('-' * 50)
print(f'Accuracy: {test_acc}')
print(f'Mis-classification: {mis_classification}')
print(f'Sensitivity: {Sensitivity}')
print(f'Precision: {Precision}')
print(f'f_1 Score: {f1score}')

cm = confusion_matrix(y, cv_results2)


def confusion_metrics(conf_matrix):
    # save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    print(f'Specificity: {conf_specificity}')
    print('-' * 50)
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)


confusion_metrics(cm)