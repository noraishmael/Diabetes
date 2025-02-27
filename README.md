# Diabetes
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
# Load the dataset
df = pd.read_csv("C:/Users/nora_/Downloads/diabetes.csv")
df.head()
#check the type of values
df.info()
# check for missing values
df.isnull().sum()
# check if there are zero values
zero_counts = (df == 0).sum()
print(zero_counts)
# in Glucose, BloodPressure, SkinThickness, Insulin, and BMI replace zero value with mean
def replace_zeros_with_mean(df, columns):
    for column in columns:
        mean_value = df[df[column] != 0][column].mean()  # Calculate mean excluding zeroes
        df[column] = df[column].replace(0, mean_value)  # Replace zeroes with mean
    return df

columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = replace_zeros_with_mean(df, columns_to_fix)
df.head(8)

# Plot histogram of the Outcome variable to see the distribution of diabetes cases
plt.hist(df["Outcome"], bins=2, edgecolor='black', alpha=0.7)
plt.xticks([0,1], ["No Diabetes", "Diabetes"])
plt.xlabel("Diabetes Outcome")
plt.ylabel("Count")
plt.title("Distribution of Diabetes in the Dataset")
plt.show()

# Scatter plot to visualize Glucose vs BMI
plt.figure(figsize=(6,4))
plt.scatter(df[df["Outcome"] == 0]["Glucose"], df[df["Outcome"] == 0]["BMI"], label="No Diabetes", alpha=0.5)
plt.scatter(df[df["Outcome"] == 1]["Glucose"], df[df["Outcome"] == 1]["BMI"], label="Diabetes", alpha=0.5, color='red')
plt.xlabel("Glucose Level")
plt.ylabel("BMI")
plt.title("Glucose vs BMI (Red = Diabetes, Blue = No Diabetes)")
plt.legend()
plt.show()

# Create a box plot to visualize the distribution of glucose levels for each outcome
plt.figure(figsize=(6,4))
df.boxplot(column="Glucose", by="Outcome", grid=True)
plt.xlabel("Diabetes Outcome")
plt.ylabel("Glucose Level")
plt.title("Box Plot of Glucose Levels by Diabetes Outcome")
plt.suptitle("")  # Remove default title
plt.show() 

# Split data into features and target variable
#Features is all the data without outcome
X = df.drop(columns=["Outcome"])  
#Target variable (0 = No Diabetes, 1 = Diabetes)
y = df["Outcome"]  

# Split into training vs testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalise the data i.e. scale for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# import tensorflow as tf
# import keras

# Build the neural network model
#Create a simple neural network with 3 layers
model = keras.Sequential()

#create input layer
model.add(keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)))

#create hidden layer
model.add(keras.layers.Dense(8, activation="relu"))

#create output layer
model.add(keras.layers.Dense(1, activation="sigmoid"))

# Compile the model i.e. how will it learn
model.compile(
    optimizer="adam",  
    loss="binary_crossentropy",  
    metrics=["accuracy"]
 ) 
history = model.fit(
    X_train,  
    y_train,  
    epochs=50, 
    batch_size=10,  
    validation_data=(X_test, y_test), 
    verbose=1)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Neural Network Training Progress')
plt.legend()
plt.show()

# Improve neural network model
model = keras.Sequential()

# Input layer: More neurons to capture patterns
model.add(keras.layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)))

# Additional hidden layer for better learning
model.add(keras.layers.Dense(16, activation="relu"))

# Output layer with sigmoid activation for binary classification
model.add(keras.layers.Dense(1, activation="sigmoid"))

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001)  # Lower learning rate for better adjustments
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,  # Increased epochs for better learning
    batch_size=16,  # Slightly larger batch size
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate model performance
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Improved Neural Network Training Progress')
plt.legend()
plt.show()

print(df["Outcome"].value_counts())

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1

print(classification_report(y_test, y_pred))

# Compute class weights to handle imbalance in the dataset
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

#Build the neural network model
model = keras.Sequential()

#Input layer
model.add(keras.layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)))

#Hidden layer
model.add(keras.layers.Dense(16, activation="relu"))

#Output layer
model.add(keras.layers.Dense(1, activation="sigmoid"))

#Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#Train the model with class weights
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,  # Apply class weights to balance learning
    verbose=1
)

#Evaluate model performance
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#Generate classification report
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred))

#Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Neural Network Training Progress with Class Weights')
plt.legend()
plt.show()

# Remove unhelpful features such as 'DiabetesPedigreeFunction' and 'SkinThickness' as they may not add much value
X = df.drop(columns=["Outcome", "DiabetesPedigreeFunction", "SkinThickness"])
y = df["Outcome"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (scaling for better performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the improved neural network model
model = keras.Sequential()

# Input layer with more neurons
model.add(keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
model.add(keras.layers.Dropout(0.3))  # Dropout to prevent overfitting

# Hidden layer
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dropout(0.3))  # Dropout layer to improve generalization

# Output layer with sigmoid activation
model.add(keras.layers.Dense(1, activation="sigmoid"))

# Compile the model with a lower learning rate for better adjustments
optimizer = keras.optimizers.Adam(learning_rate=0.0005)  # Lower learning rate for stability
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model with improved parameters
history = model.fit(
    X_train,
    y_train,
    epochs=150,  # Increased epochs for better learning
    batch_size=32,  # Larger batch size for stability
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate model performance
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Neural Network Training Progress with Feature Selection & Dropout')
plt.legend()
plt.show()

