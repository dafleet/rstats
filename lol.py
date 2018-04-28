
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('lol.csv')
df.describe()


# In[2]:


#delete 0 columns and obviously correlated ones
df = df[df['num_subscribers'] != 0]
df.drop(['views','ViewRate_Binary','num_subscribers','id'], axis=1, inplace=True)


# In[3]:


#encode categorical vars
cols = list(['influencer',
 'dow',
 'has_tags',
 'prod_focus',
 'reach',
 'media_type',
 'post_type',
 'influencer_type',
 'FTC_Converted',
 'In_Person_Converted',
 'Giveaway_Converted',
 'Reoccuring_Converted'])
lol = pd.get_dummies(df, prefix=cols,columns=cols,drop_first=True)
X = lol.loc[:, lol.columns != 'view_rate']
X.head(2)


# In[4]:


#encode y
y = lol['view_rate']
y_cat = pd.qcut(lol['view_rate'], 4, labels=["25%", "50%", "75%", "100%"])
y = pd.concat([y,y_cat], axis=1)
y.columns=['view_rate','view_rate_pctl']
y.head(2)


# In[5]:


# SVM classifier
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train2, y_test2 = train_test_split(X,y,test_size=0.2,random_state=21)

# Use the categorical response
y_train, y_test = y_train2['view_rate_pctl'], y_test2['view_rate_pctl']

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline,parameters,cv=5)

# Fit to the training set
cv.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy for SVM: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


# In[6]:


from sklearn.ensemble import RandomForestClassifier

steps = [('scaler', StandardScaler()),
         ('RF', RandomForestClassifier())]

pipeline = Pipeline(steps)

param_grid = { 
    'RF__n_estimators': [200, 500],
    'RF__max_features': ['auto', 'sqrt', 'log2'],
    'RF__max_depth' : [4,5,6,7,8],
    'RF__criterion' :['gini', 'entropy']
}

clf = GridSearchCV(pipeline,param_grid,cv= 5)
clf.fit(X_train, y_train)


# In[7]:


y_pred=clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(clf.best_params_))


# In[8]:


import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y['view_rate_pctl'])
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[9]:


# Kth nearest neighbors 
from sklearn.neighbors import KNeighborsClassifier
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train,y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))


# In[10]:


# # Recursive Feature Elimination
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression

# # create a base classifier used to evaluate a subset of attributes
# logm = LogisticRegression()
# # create the RFE model and select 3 attributes
# rfe = RFE(logm, 3)
# rfe = rfe.fit(X, y['view_rate_pctl'])
# # summarize the selection of the attributes
# features = rfe.support_

# for idx in np.where(features)[0]:
#     print(list(X)[idx])

#print(df.groupby(['influencer'])['view_rate'].aggregate('median'))


# In[11]:


# from sklearn.linear_model import ElasticNet

# steps = [#('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
#          ('scaler', StandardScaler()),
#          ('elasticnet', ElasticNet())]

# # Create the pipeline: pipeline 
# pipeline = Pipeline(steps)

# # Specify the hyperparameter space
# parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# # Use the continuous var
# y_train, y_test =  y_train2['view_rate'], y_test2['view_rate']

# # Create the GridSearchCV object: gm_cv
# gm_cv = GridSearchCV(pipeline,parameters,cv=5)

# # Fit to the training set
# gm_cv.fit(X_train,y_train)

# # Compute and print the metrics
# r2 = gm_cv.score(X_test, y_test)
# print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
# print("Tuned ElasticNet R squared: {}".format(r2))

