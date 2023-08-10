import tensorflow as tf
#from tensorflow.keras.preprocessing import image
import numpy as np

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_model():

    # Load the Wine dataset
    data = load_wine()
    X = data.data
    y = data.target


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = training_model(X_train_scaled, y_train)
    return model


    # Initialize the KNN classifier
    

    # Train the model
    


    
def training_model(X_train_scaled, y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    return knn

# def predict(model, image_path):
#     img = image.load_img(image_path, target_size=(224, 224))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
#     predictions = model.predict(img)
    
#     # Decode predictions (this is a simplified example)
#     decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

#     return decoded_predictions



def predict(model, input_data):
#    load_model()
    input_data = [1.423e+01, 1.710e+00, 2.430e+00, 1.560e+01, 1.270e+02, 2.800e+00,
       3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00,
       1.065e+03]
    # Preprocess the input data using the same scaler as in training
#    scaled_input = scaler.transform(np.array(input_data).reshape(1, -1))
    
    # Make predictions
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    
    return prediction
