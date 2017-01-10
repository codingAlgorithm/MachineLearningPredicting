#author: Janney
# panda_play.py
# Use pandas on Titanic data to predict survival.
# Generates random_forest model, and model based on survival probabilities
# binned by class,gender, and fare.
# Outputs prediction data with Passengerid, and survival.

# Read in data from csv.
# Then explore with summaries and figures.

import csv   # load csv reader library
import pandas as pd  # load python datascience library.
import numpy as np  # load numpy
from sklearn.linear_model import LogisticRegression


# Set random seed
np.random.seed(1020304)




# Function to clean data in specified data frame, and return numpy array.
# Returns cleaned values, and column names.
def clean_data(df_in):
    df_in['Gender'] = df_in['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df_in['EmbarkedN'] = df_in['Embarked'].dropna().map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    # replace missing ages with median value for that class
    median_ages = np.zeros((2, 3))
    median_embark = np.zeros((2, 3))
    # make copy of column
    df_in['Agefill'] = df_in['Age']

    # set maximum fare to 40.
    df_in.Fare[df_in.Fare > 40] = 40

    # Replace Missing ages, Embarkation, and Fare info with median for the gender, and class
    for i in range(0, 2):  # gender
        for j in range(0, 3):  # class
            # fix missing Ages
            median_ages[i, j] = df_in[(df_in['Gender'] == i) & (df_in['Pclass'] == j + 1)]['Age'].dropna().median()

            df_in.loc[(df_in.Gender == i) & (df_in.Pclass == j + 1) & (df_in.Age.isnull()), 'Agefill'] = median_ages[
                i, j]

            # fix missing embarkation
            median_embark[i, j] = df_in[(df_in['Gender'] == i) & (df_in['Pclass'] == j + 1)][
                'EmbarkedN'].dropna().median()
            df_in.loc[(df_in.Gender == i) & (df_in.Pclass == j + 1) & (df_in.EmbarkedN.isnull()), 'EmbarkedN'] = \
            median_embark[i, j]

            # fix missing/zero fare
            median_fare = df_in[(df_in['Gender'] == i) & (df_in['Pclass'] == j + 1)]['Fare'].dropna().median()
            df_in.loc[(df_in.Gender == i) & (df_in.Pclass == j + 1) & \
                      (df_in.Fare.isnull() | df_in.Fare == 0 | np.isnan(df_in.Fare)),
                      'Fare'] = median_fare

    # Feature engineering: Nonlinear combinations of columns.

    # create family size column
    df_in['FamilySize'] = df_in.Parch + df_in.SibSp

    # create AgeClass column
    df_in['AgeClass'] = df_in.Agefill * df_in.Pclass

    # first attempt. Got 0.74
    df_in = df_in.drop(['PassengerId', 'Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    df_in = df_in.drop(['FamilySize', 'Parch', 'AgeClass', 'SibSp', 'EmbarkedN'], axis=1)
    return df_in, list(df_in)


##Random forest classifier
forest = LogisticRegression(n_jobs=3)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
output = forest.predict(test_data)

prediction_file = open('output/logisticregression.csv', 'wt')
prediction_file_object = csv.writer(prediction_file)

# read in as data frame
df = pd.read_csv('/Users/Janney/Downloads/test.csv', header=0)
train_df = df.copy()
[train_df, train_header] = clean_data(train_df)
train_data = train_df.values

df_t = pd.read_csv('/Users/Janney/Downloads/test.csv', header=0)
test_df = df_t.copy()
[test_df, test_header] = clean_data(test_df)
test_data = test_df.values

# output prediction from random forest model
prediction_file_object.writerow(["PassengerId", "Survived"])
for j in range(0, len(output)):
    prediction_file_object.writerow([df_t.PassengerId[j].astype(int), output[j].astype(int)])
prediction_file.close()