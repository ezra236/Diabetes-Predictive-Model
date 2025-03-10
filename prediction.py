#Importing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#Loading Dataset
df = pd.read_csv('diabetes.csv')

# Showing 5 rows from the dataset
df.head()

# Splitting the data into training and testing sets
X = df.drop('Outcome', axis = 1)
y = df['Outcome']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)

#standardizing the data
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

#Create the model
model = Sequential([
    Dense(32, activation = 'relu', input_shape = (xtrain.shape[1],)),
    Dropout(0.1),
    Dense(32, activation = 'relu'),
    Dropout(0.5),
    Dense(1, activation = 'sigmoid')
])

#Model Compliation 
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

#Training the model
model.fit(xtrain, ytrain, epochs = 20, batch_size = 16, validation_data = (xtest, ytest))

# Model Results
loss, accuracy = model.evaluate(xtest, ytest)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Save the trained model
model.save('pima_diabetes_model.h5')

#loading Saved Model 
from tensorflow.keras.models import load_model
loaded_model = load_model('pima_diabetes_model.h5')

#Predicting with new Data 
# Example new data: [pregnancies, glucose, bloodpressure, skinthickness, insulin, BMI, DPF, age]
new_data = [[5, 114, 60, 19, 120, 25.8, 0.587, 21]]

# Preprocess the new data (scale it)
new_data_scaled = scaler.transform(new_data)

# Predict using the loaded model
prediction = loaded_model.predict(new_data_scaled)

# Convert the output from probability to class (0 or 1)
predicted_class = (prediction > 0.5).astype(int)

print(f'Prediction: {predicted_class[0][0]}')

