
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a SVM classifier using scikit-learn
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Train a neural network using TensorFlow/Keras
model = Sequential([
    Dense(8, input_dim=4, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)


def predict_svm():
    input_values = [float(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get())]
    scaled_input = scaler.transform([input_values])
    svm_prediction = svm_model.predict(scaled_input)
    svm_result.set(f"SVM Prediction: {svm_prediction[0]}")

def predict_neural_network():
    input_values = [float(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get())]
    scaled_input = scaler.transform([input_values])
    nn_prediction = model.predict_classes(scaled_input)
    nn_result.set(f"Neural Network Prediction: {nn_prediction[0]}")


root = tk.Tk()
root.title("AI Classification GUI")


label1 = ttk.Label(root, text="Feature 1:")
label1.grid(row=0, column=0, padx=10, pady=5)
entry1 = ttk.Entry(root)
entry1.grid(row=0, column=1, padx=10, pady=5)

label2 = ttk.Label(root, text="Feature 2:")
label2.grid(row=1, column=0, padx=10, pady=5)
entry2 = ttk.Entry(root)
entry2.grid(row=1, column=1, padx=10, pady=5)

label3 = ttk.Label(root, text="Feature 3:")
label3.grid(row=2, column=0, padx=10, pady=5)
entry3 = ttk.Entry(root)
entry3.grid(row=2, column=1, padx=10, pady=5)

label4 = ttk.Label(root, text="Feature 4:")
label4.grid(row=3, column=0, padx=10, pady=5)
entry4 = ttk.Entry(root)
entry4.grid(row=3, column=1, padx=10, pady=5)


svm_result = tk.StringVar()
nn_result = tk.StringVar()

svm_button = ttk.Button(root, text="SVM Predict", command=predict_svm)
svm_button.grid(row=4, column=0, columnspan=2, pady=10)
svm_result_label = ttk.Label(root, textvariable=svm_result)
svm_result_label.grid(row=5, column=0, columnspan=2, pady=10)

nn_button = ttk.Button(root, text="Neural Network Predict", command=predict_neural_network)
nn_button.grid(row=6, column=0, columnspan=2, pady=10)
nn_result_label = ttk.Label(root, textvariable=nn_result)
nn_result_label.grid(row=7, column=0, columnspan=2, pady=10)


root.mainloop()
