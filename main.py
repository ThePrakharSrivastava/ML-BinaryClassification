import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# load data from dataset file
my_data = pd.read_csv('C:\Projects\ML-BinaryClassification\iris\Iris.csv')

#split the dataset into features and labels
features = my_data.iloc[:, : 5]

labels = my_data[my_data.columns[-1]]

# Split our data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, random_state=42)

#print(test_labels)
#print (features)
#print (labels)

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)

# Make predictions
print(test)
preds = gnb.predict(test)
print(preds)

# Evaluate accuracy
print(accuracy_score(test_labels, preds))

