import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical

"""
This project use a dataset from Kaggle to predict the survival of 
patients with heart failure from serum creatinine and ejection fraction,
and other factors such as age, anemia, diabetes, and so on.

A model for predicting mortality caused by heart failure
can be of great help for early detection and management

dataset: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
"""

# load dataset
data = pd.read_csv('heart_failure.csv')

# data inspection
print(data.info())

# column to predict
print('Classes and values in the dataset',Counter(data['death_event']))

# asign column for variable data and labels
y = data['death_event']
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

# one-hot encoding
x = pd.get_dummies(x)

# split datasets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# scale numeric features
cols = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
ct = ColumnTransformer([("numeric", StandardScaler(), cols)])

# fit to dataset
X_train = ct.fit_transform(X_train)
X_test = ct.fit_transform(X_test)

# Initialize an instance of LabelEncoder
le = LabelEncoder()
Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.fit_transform(Y_test.astype(str))

# transform the encoded training labels Y_train into a binary vector
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# design the model
model = Sequential()
model.add(InputLayer(X_train.shape[1]))
model.add(Dense(12, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training model
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)

# evaluate model
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print("Loss: ", loss, "Accuracy: ", acc)

# prediction using the model
y_estimate = model.predict(X_test, verbose=0)
# method to select the indices
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(Y_test, axis=1)
# prediction results
print(classification_report(y_true, y_estimate))