import pandas
# Import the linear regression class
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
import numpy as np
# Sklearn also has a helper that makes it easy to do cross-validation
from sklearn.cross_validation import KFold
import csv



titanic_df = pandas.read_csv("data/train.csv")
# print(titanic_df.describe())
# gives a lot of insight into the data
titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())
# fills missing age with the median <3

# axis - 0 means drop across each column, 1 means drop across each row
# we remove columns when they have irreleavant data
# removes the labels for ticket and cabin because we removed their data
titanic_df = titanic_df.drop(['Ticket', 'Cabin'], axis=1)

# If any row has any NaN value it is removed
# titanic_df = titanic_df.dropna()

# we want to replace all male to 0 and female to 1
# this code can be used to select all the fields that match the criteria mentioned
titanic_df.loc[titanic_df["Sex"] == "male", "Sex"] = 0
titanic_df.loc[titanic_df["Sex"] == "female", "Sex"] = 1

# Since the most common gate was S all the ones with S have been marked with S
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# S, C, Q have been converted to numbers again
titanic_df.loc[titanic_df["Embarked"] == "S", "Embarked"] = 0
titanic_df.loc[titanic_df["Embarked"] == "C", "Embarked"] = 1
titanic_df.loc[titanic_df["Embarked"] == "Q", "Embarked"] = 2

# cleaning done

# prediction code using the test data
#-----------------------#-----------------------#-----------------------#-----------------------#-----------------------
# # this code splits the test data into three parts and checks how well our are guesses
# # The columns we'll use to predict the target
# predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#
# # Initialize our algorithm class
# alg = LogisticRegression(random_state=1)
#
# # Compute the accuracy score for all the cross-validation folds; this is much simpler than what we did before
# # basically first field is the algorithm, second one is the features and third is the result,
# scores = cross_validation.cross_val_score(alg, titanic_df[predictors], titanic_df["Survived"], cv=3)
# # Take the mean of the scores (because we have one for each fold)
# print(scores.mean())
#-----------------------#-----------------------#------#-----------------------#-----------------------#-----------------------

# Adding predictions to the final file
titanic_test = pandas.read_csv("data/test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_df["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
# Since the most common gate was S all the ones with S have been marked with S
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

# S, C, Q have been converted to numbers again
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic_df[predictors], titanic_df["Survived"])

# Make predictions using the test set
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the data set
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
# print submission
submission.to_csv(path_or_buf="prediction.csv")
# submitted ! 75.12 % accuracy