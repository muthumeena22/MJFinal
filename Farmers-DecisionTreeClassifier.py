import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
le=preprocessing.LabelEncoder()

xtrain_job1 = pd.read_csv(r'C:\Users\HP\Desktop\hcl_hackathon_event\crop_production.csv')
xtrain_job1.isnull().sum()#to find missing values



print("Dataset Lenght:: ", len(xtrain_job1))
print("Dataset Shape:: ", xtrain_job1.shape)


"""imputer=Imputer(strategy='mean',missing_values='NaN',axis=0)
imputer=imputer.fit(xtrain_job[:,5])
xtrain_job[:,5]=imputer.transform(xtrain_job[:,5])"""

xtrain_job1['Production'].fillna(xtrain_job1['Production'].mean(),inplace=True)#to resolve missing values

xtrain_class=xtrain_job1['Crop']

for column in xtrain_job1.columns:
    if xtrain_job1[column].dtype == type(object):
        xtrain_job1[column] = le.fit_transform(xtrain_job1[column])

train=xtrain_job1.values[:]
print(train)
        
        
classvar=xtrain_class
print(classvar)

#slicing data for training and testing
X_train, X_test, y_train, y_test = train_test_split( train, classvar, test_size = 0.4, random_state = 100)
print(X_train)
#decision tree with gini index
"""clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=5, min_samples_leaf=10)
clf_gini.fit(X_train, y_train)
y_predgini = clf_gini.predict(X_test)
print(y_predgini)"""


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 1000,max_depth=15, min_samples_leaf=20)
clf_entropy.fit(X_train, y_train)
y_predentropy=clf_entropy.predict(X_test)
print(y_predentropy)


#accuracy
#acc_gini=accuracy_score(y_test,y_predgini)*100
#print("Accuracy-gini",acc_gini)
acc_entropy=accuracy_score(y_test,y_predentropy)*100
print("Accuracy-entropy",acc_entropy)

