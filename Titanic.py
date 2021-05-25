#Collecting Data

import pandas as pd 
# used for data analysis
import numpy as np 
# this library is used for scientific computation
import seaborn as sns 
# used for statistical plotting
import matplotlib.pyplot as plt
# to run this library in jupiter notebook
import math 

titanic_data = pd.read_csv("Titanic.csv")
data=titanic_data.head(10)
print(data)


print("no. of passengers in original data:" + str(len(titanic_data.index)))

#How many survived and how many people did not survive
sns.countplot(x="Survived", data=titanic_data)

# How many people of which class survived

sns.countplot(x="Survived", hue="Pclass", data=titanic_data)

# Age plot

titanic_data["Age"].plot.hist()

titanic_data["Fare"].plot.hist(bins=20, figsize=(10, 5))


# this will basically tell us what all values are null
Null_values=titanic_data.isnull()
print(f"Null values :",Null_values)

Null_sum = titanic_data.isnull().sum()
print(f"Sum of null values : ", Null_sum)

# We'll analysis missing values with help of heat map

sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap="viridis")

# Dropping Cabin coloumn coz many values are nan 

titanic_data.drop("Cabin", axis=1, inplace=True)

titanic_data.head(5)

# Dropping all nan value from data 

titanic_data.dropna(inplace=True)

titanic_data.isnull().sum()

# we have many string values so this has to bhi converted into Categorical variable in order to
# perform logitic regression(beacuse it only takes two values 0,1)
# we will convert Categorical variables into some dummy variables using pandas

sex = pd.get_dummies(titanic_data['Sex'], drop_first=True)
sex.head(5)

embarked = pd.get_dummies(titanic_data["Embarked"])
embarked.head(5)

embarked = pd.get_dummies(titanic_data["Embarked"], drop_first= True)
embarked.head(5)

Pcl = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
Pcl.head(5)

# Now we'll concatinate all these new rows to our data set

titanic_data = pd.concat([titanic_data, sex, embarked, Pcl], axis=1)
titanic_data.head(5)

# Now we'll drop the columns which are haveong string value like Name, Sex, Ticket and the columns which are not useful to us like PassengerId

titanic_data.drop(['Sex', 'Embarked', 'PassengerId', 'Name', 'Ticket'],axis=1,inplace=True)
titanic_data.head(5)

# titanic_data.drop('Pclass', axis=1, inplace= True)
titanic_data.head(5)

# Here we'll split the data into train subset and test subset.
# then we'll build a model on train data and predict the ouput on test data set

# Dependent variable (we have the discrete outcome)
X = titanic_data.drop("Survived", axis=1)
# Independent variable (value data we need to predict)
y = titanic_data["Survived"]

# Now for splittibg data into testing and training subset we'll be using sklearn

from sklearn.model_selection import train_test_split

# type `train_test_split` and press shift+tab and you will able to see example of how train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# import Logistic Regression

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='lbfgs', max_iter=891)

logmodel.fit(X_train,y_train)

print(f"y_train : ",y_train)
predictions = logmodel.predict(X_test)

# Now we'll evaluate how our model has been performing
# we can calculate the accuraccy or we can calculate the classification report

from sklearn.metrics import classification_report

report = classification_report(y_test, predictions)
print(f"Classification report : ",report)
print(f"y_test : ",y_test)

from sklearn.metrics import confusion_matrix
y = confusion_matrix(y_test,predictions)
print(f"Confusion matrix : ", y)

from sklearn.metrics import accuracy_score

x = accuracy_score(y_test, predictions)*100
print(f"Accuracy : ", x)