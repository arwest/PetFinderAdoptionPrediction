#!/usr/bin/env python
# coding: utf-8

# In[29]:


import json
import random
import os
import gc
import html
import time
import re
from collections import defaultdict
import math

import numpy as np
import pandas as pd
import scipy
from scipy import ndimage
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score, mean_squared_error, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from nltk.stem.snowball import SnowballStemmer, PorterStemmer



import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', UserWarning)
get_ipython().run_line_magic('matplotlib', 'inline')

# treat np.inf as nan value too, because machine learning algorithms should not get inf as input
pd.set_option("use_inf_as_na", True)
tqdm.pandas()


# # PROBLEM
# 
# import csv
# import json
# import os
# from os.path import isfile, join
# 
# HA = "\\home.iowa.uiowa.edu\vtiema\Desktop\petfinder-adoption-prediction"
# HA2 = "\\home.iowa.uiowa.edu\vtiema\Desktop\petfinder-adoption-prediction\train_sentiment"
# 
# desc_index = 20
# pet_id_index = -3
# 
# def create_new_train_csv():
#     sent = dict()
# 
#     json_folder_dir = join(os.curdir, HA , HA2)
# 
#     for file in os.listdir(json_folder_dir):
#         name = file.split('.')[0]
#         if isfile(join(json_folder_dir, file)):
#             with open(join(json_folder_dir, file)) as json_file:
#                 data = json.load(json_file)
#                 doc_sent = data['documentSentiment']
#                 sent[name] = (doc_sent['magnitude'], doc_sent['score'])
# 
#     data = []
#     with open(join(os.curdir, HA, 'train.csv')) as csvFile:
#         csvReader = csv.reader(csvFile, delimiter=',')
#         line = 0
#         for row in csvReader:
#             if line == 0:
#                 headers = row[:20] + ['DescMagnitude', 'DescScore'] + row[-3:]
#                 line += 1
#                 continue
#             try:
#                 (magnitude, score) = sent[row[pet_id_index]]
#             except KeyError:
#                 magnitude = score = 'NaN'
#             new_row_data = row[:20] + [magnitude, score] + row[-3:]
#             data.append(new_row_data)
# 
#     with open(join(os.curdir, HA , 'train_wo_desc.csv'), 'w', newline='') as csvFile:
#         csvWriter = csv.writer(csvFile, delimiter=',')
#         csvWriter.writerow(headers)
#         for data_line in data:
#             csvWriter.writerow(data_line)
# 
#     return
# 
# create_new_train_csv()

# In[114]:


import pandas as pd

train = pd.read_csv("train.csv")

train = train.drop(["RescuerID","VideoAmt","PetID", "PhotoAmt","Description"], axis = 1)


# ### MISSING VALUES

# In[85]:


train.isna().sum()


# ### Missing values in the name columns

# In[86]:


train['Name'] = train[['Name']].fillna(value= 0)
train['hasName'] = (train['Name'] != 0).astype('int64')
train = train.drop('Name', axis=1)
train.head()


# In[87]:


train.columns


# ### One hot encoding of the categorical Values

# # creating dummy variables
# Type = pd.get_dummies(train["Type"], prefix = "Type")
# breed1 = pd.get_dummies(train["Breed1"], prefix = 'Breed1')
# breed2 = pd.get_dummies(train["Breed2"], prefix = 'Breed2')
# gender = pd.get_dummies(train["Gender"], prefix = "Gender")
# color1 = pd.get_dummies(train["Color1"], prefix = "Color1")
# color2 = pd.get_dummies(train["Color2"], prefix = "Color2")
# color3 = pd.get_dummies(train["Color3"], prefix = "Color3")
# maturity = pd.get_dummies(train["MaturitySize"], prefix = "Mature")
# FurLength = pd.get_dummies(train["FurLength"], prefix = "Fur")
# Vaccinated = pd.get_dummies(train["Vaccinated"], prefix = "Vaccinated")
# Dewormed = pd.get_dummies(train["Dewormed"], prefix = "Dewormed")
# Sterilized = pd.get_dummies(train["Sterilized"], prefix = "Sterilized")
# Health = pd.get_dummies(train["Health"], prefix = "Health")
# State = pd.get_dummies(train["State"], prefix = "State")
# 
# 
# 
# #adding the new columns to the dataframe
# train = pd.concat( [train, Type, breed1,breed2, gender,color1, color2, color3, Vaccinated,
#                     Dewormed, Sterilized, Health,State,maturity,FurLength ] , axis = 1)
# 
# y = train['AdoptionSpeed'].values    #returns a numpy array
# 
# 
# #dropping old feature columns
# train = train.drop(["Type", "Breed1","Breed2","Gender", "Color1","Color2", "Vaccinated", "Dewormed", 
#                     "Sterilized", "Health","Color3","MaturitySize","FurLength","State", 'AdoptionSpeed'], axis = 1)
# 
# 
# train.head()

# In[ ]:





# In[116]:


# more efficient way of One Hot encoding the 

col = ['Color1','Color2','Health', 'Gender', 'Dewormed','Type','MaturitySize', 'Sterilized','Vaccinated','FurLength']
for i in col:
    train = pd.concat([train.drop(i, axis=1),pd.get_dummies(train[i], prefix=i)], axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[94]:


# standardizing the data

from sklearn import preprocessing

x = train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)


# In[97]:


# splitting testing and training data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.33, random_state = 6)


# ## Training and Testing

# In[104]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, log_loss


clf1 = DecisionTreeClassifier(criterion='gini',splitter='best', random_state=6, max_leaf_nodes=22)
clf1.fit(X_train, y_train)
y_pred1 = clf.predict(X_test)

print("accuracy score is:", accuracy_score(y_test,y_pred1))

cohen_kappa_score(y_pred1, y_test, weights = "quadratic")


# In[105]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss


clf2 = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=6)
rf_model = clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

print("accuracy score is:", accuracy_score(y_test,y_pred2))
y_prob = rf_model.predict_proba(X_test)
print("the log loss is:",log_loss(y_test, y_prob))


# In[118]:


import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

Cs = [0.1, 1, 10, 100, 1000]

clf = LogisticRegressionCV(Cs, cv = 5, random_state = 0).fit(X_train, y_train)

scores = clf.scores_
Accuracy = []


for i in scores.values():
    for j in i:
        best = max(j)
        Accuracy.append(best)
        
max_Accuracy = max(Accuracy)

for i in range(len(Accuracy)):
    if Accuracy[i] == max_Accuracy:
        print("The best C value is", Cs[i],"with accuracy",max_Accuracy * 100)
        


# In[ ]:




