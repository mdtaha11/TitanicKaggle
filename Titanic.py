# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 18:29:41 2020

@author: Taha
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

def subs_in_substrings(big_string,substrings):
    for substring in substrings:
        if big_string.find(substring)!=-1:
            return substring
    print(substring)
    return np.nan  

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
 
train['Titles']=train['Name'].map(lambda x:subs_in_substrings(x,title_list))
test['Titles']=test['Name'].map(lambda x:subs_in_substrings(x,title_list))

def titleChange(data):
    title=data['Titles']
    if(title in ['Countess', 'Mme']):
        return('Mrs')
    elif title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return('Mr')
    elif title in ['Mlle', 'Ms','Miss']:
        return('Miss')
    elif title=='Dr':
        if(data['Sex']=='Male'):
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
train['Titles']=train.apply(titleChange,axis=1)
test['Titles']=test.apply(titleChange,axis=1)

train  = train.drop(['Name'], axis=1)
test  = test.drop(['Name'], axis=1)
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
train['Titles']=train['Titles'].map(titles)
test['Titles']=test['Titles'].map(titles)
train['Titles'].value_counts()

plt.figure(figsize=(15,5))
sns.countplot(x='Titles',hue='Survived',data=train)

plt.figure(figsize=(15,5))
sns.countplot(x='SibSp',hue='Survived',data=train)
 
corr=train.corr()
sns.heatmap(corr,annot=True)

train.isnull().sum()

survived='survived'
not_survived='not survived'

fig,axes=plt.subplots(nrows=1,ncols=2)
female=train[train['Sex']=='female']
male=train[train['Sex']=='male']
ax=sns.distplot(female[female['Survived']==1].Age.dropna(),bins=18, label = survived, 
                ax = axes[0], kde =False)
ax=sns.distplot(female[female['Survived']==0].Age.dropna(),bins=40, label = not_survived, 
                ax = axes[0], kde =False)
ax=sns.distplot(male[male['Survived']==1].Age.dropna(),bins=18, label = survived, 
                ax = axes[1], kde =False)
ax=sns.distplot(male[male['Survived']==0].Age.dropna(),bins=40, label = not_survived, 
                ax = axes[1], kde =False)

female['Survived'].value_counts()
male['Survived'].value_counts()

grid=sns.FacetGrid(train,col="Embarked",size=3.2,aspect=1.6)
grid.map(sns.pointplot,'Pclass','Survived','Sex')
grid.add_legend()

#Pclass
grid=sns.FacetGrid(train,col="Survived",size=3.2,aspect=1.6)
grid.map(sns.countplot,'Pclass')
grid.add_legend()

sns.barplot(x='Pclass',y='Survived',data=train)

grid=sns.FacetGrid(train,col='Pclass',row='Survived',height=3.2,aspect=1.6)
grid.map(plt.hist,'Age')
grid.add_legend()

#SibSp and Parch
data=[train,test]
for dataset in data:
    dataset['Relatives']=dataset['SibSp']+dataset['Parch']
    dataset.loc[train['Relatives']>0,'not_alone']=1
    dataset.loc[train['Relatives']==0,'not_alone']=0

sns.barplot(x='Relatives',y='Survived',data=train)
   
#Data preprocessing

#No use of ID 
train=train.drop('PassengerId',axis=1)

#Dealing with the column 'Cabin'
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6,"T":9, "G": 7, "U": 8,
        'X':0}
train.isnull().sum()
data=[train,test]
for dataset in data:
    dataset['Cabin']=dataset['Cabin'].fillna('X')  
    dataset['CabinNumber']=[c[0] for c in dataset['Cabin']]
train['CabinNumber']=train['CabinNumber'].map(deck)
test['CabinNumber']=test['CabinNumber'].map(deck)

#Drop the cabin feature
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)    

test.isnull().sum()
train.isnull().sum()

#Dealing with missing values
train=train.interpolate()   
test=test.interpolate()   

#Replacing with most frequent value in Embarked
train.Embarked.describe() 
 
data=[train,test]
for dataset in data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
    
#Categorical
data = [train, test]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
    
genders = {"male": 0, "female": 1}
data = [train, test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders) 
    
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)    

ports = {"S": 0, "C": 1, "Q": 2}
data = [train , test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
    
data=[train,test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[dataset['Age']<=14,'Age']=0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 21), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 21) & (dataset['Age'] <= 25), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 29), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 35), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 43), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 43) & (dataset['Age'] <= 68), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 68, 'Age'] = 6
    
train['Age'].value_counts()

 

data = [train, test]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

#Adding new features
data = [train , test ]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
    
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['Relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
    
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
    
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)    
classifier.fit(X_train,Y_train)   

Y_pred=classifier.predict(X_test)    
classifier.score(X_train,Y_train)  
accuracy=round(classifier.score(X_train,Y_train)*100,2)    
    
    
    
    
    
    