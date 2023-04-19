# -*- coding: utf-8 -*-

# 1. Import packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import joblib
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# 2. Load data from csv
data = pd.read_csv('C:/Users/ivanz/OneDrive/Desktop/ML/heart.csv')
data.head()

# Check data types
print(data.dtypes)

# 3. Check for missing values
data.isna().sum()

labels = data['target']
features = data.drop('target', axis=1)
print("Heart Disease dataset has {} data points with {} variables each.".format(*data.shape))

print('===================Features===================')
print(features.head())
print('===================Labels===================')
print(labels.head())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Check data types of X_train and y_train
print(type(X_train))
print(type(y_train))


# Create sequential network using keras
nClass = len(np.unique(y_train))

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(nClass, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy', metrics=['accuracy'])


# EarlyStopping
es = EarlyStopping(patience=10, verbose=1, restore_best_weights=True)

# TensorBoard
PATH = "tb_logs"
LOG_PATH = os.path.join(PATH, 'heart_disease',
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=LOG_PATH)

# Model training
BATCH_SIZE = 32
EPOCHS = 100
history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[es, tb])

# Model Evaluation
# Train evaluation loss and accuracy
print(f"Train Evaluation : \n { model.evaluate(X_train,y_train)}")

# Test evaluation loss and accuracy
print(f"Test Evaluation : \n { model.evaluate(X_test,y_test)}")

# Model predictions
threshold = 0.5
predictions = (model.predict(X_test) >= threshold).astype(int)
print("Predictions:", predictions)

# Confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print("Confusion Matrix:")
print(cm)

# Pickle model and columns
joblib.dump(model, 'C:/Users/ivanz/OneDrive/Desktop/ML//heart_disease_model.pkl')
joblib.dump(features.columns, 'C:/Users/ivanz/OneDrive/Desktop/ML//heart_disease_columns.pkl')