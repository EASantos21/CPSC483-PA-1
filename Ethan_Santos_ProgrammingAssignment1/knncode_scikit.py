# Importing necessary libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Importing the csv file
df = pd.read_csv(r'HappinessData-1.csv')
df = df.dropna()

# Splitting the data set
X_train, X_test, y_train, y_test = train_test_split(df, df['Unhappy/Happy'], test_size = 0.2)

# Predicting and printing out the accuracy
KNN_Model = KNeighborsClassifier(n_neighbors = 11)
KNN_Model.fit(X_train, y_train)
testPred = KNN_Model.predict(X_test)

print(metrics.accuracy_score(y_test, testPred))