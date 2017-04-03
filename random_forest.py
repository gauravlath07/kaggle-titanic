import pandas
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import re

def get_title_code(name):
    start_index = name.index(',')
    start_index += 2
    end_index = name.index('.')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                     "Mme": 8, "Don": 9, "Lady": 10, "the Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona":10}
    title = name[start_index:end_index]
    return title_mapping[title]


titanic_df = pandas.read_csv("data/train.csv")
titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].median())
titanic_df = titanic_df.drop(['Ticket', 'Cabin'], axis=1)
titanic_df.loc[titanic_df["Sex"] == "male", "Sex"] = 0
titanic_df.loc[titanic_df["Sex"] == "female", "Sex"] = 1
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
titanic_df.loc[titanic_df["Embarked"] == "S", "Embarked"] = 0
titanic_df.loc[titanic_df["Embarked"] == "C", "Embarked"] = 1
titanic_df.loc[titanic_df["Embarked"] == "Q", "Embarked"] = 2

# Creating new features to improve accuracy
titanic_df["total_family"] = titanic_df["SibSp"]+titanic_df["Parch"]
titanic_df["name_length"] = titanic_df["Name"].str.len()

# Get all of the titles
titanic_df['titles'] = titanic_df["Name"].apply(get_title_code)

# cleaning done



# predictors = ["Pclass", "Sex", "Fare", "name_length", "titles"]

# Perform feature selection, gives the top 5 feauteres
# selector = SelectKBest(f_classif, k=5)
# selector.fit(titanic_df[predictors], titanic_df["Survived"])
# scores = -np.log10(selector.pvalues_)
# print scores
# print scores adnd look at the top scores

#------------------------------------------------------------------------------------------------------------
# alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
#
# # Compute the accuracy score for all of the cross validation folds; this is much simpler than what we did before
# kf = cross_validation.KFold(titanic_df.shape[0], n_folds=3, random_state=1)
# scores = cross_validation.cross_val_score(alg, titanic_df[predictors], titanic_df["Survived"], cv=kf)
#
# # Take the mean of the scores (because we have one for each fold)
# print(scores.mean())
# for testing within the data set
#------------------------------------------------------------------------------------------------------------

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
# Creating new features to improve accuracy
titanic_test["total_family"] = titanic_test["SibSp"]+titanic_test["Parch"]
titanic_test["name_length"] = titanic_test["Name"].str.len()

# Get all of the titles
titanic_test['titles'] = titanic_test["Name"].apply(get_title_code)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "name_length", "total_family", "titles"]
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

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



