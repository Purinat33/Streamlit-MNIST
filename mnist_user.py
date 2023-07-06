import streamlit as st
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import urllib.request
from keras.datasets import mnist
import random
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import os

st.title("MNIST Handwritten Digit Prediction")
st.write("Now you can run MNIST yourself!")
st.divider()

# IMPORTING THE MODEL
# Path starts at ~/users/user_name for some reasom
# Update: File path for opening in Streamlit and for running in normal python are DIFFERENT

# Load locally
# filepath = os.path.abspath("./Desktop/youtube/streamlit/my_mnist.h5")
# model = keras.models.load_model(filepath)

# Load from GitHub
# Note that the file URL needs to be for the `raw` file
url = "https://github.com/Purinat33/streamlit_stock_price/raw/master/my_mnist.h5"
file_path = keras.utils.get_file("my_mnist.h5", origin=url)
model = keras.models.load_model(file_path)
# Load image
img_url = "https://github.com/Purinat33/streamlit_stock_price/raw/master/mnist_overview_95.png"
st.header("Model Architectural Overview")
urllib.request.urlretrieve(img_url, "mnist_overview_95.png")
img = Image.open("mnist_overview_95.png")
st.image(img)


# Model Metrics
def getImage(label, y_test):
    breakPoint = 0
    while True:
        rand = random.randint(0, 9999)
        if y_test[rand] == label:  # If the index = the label we want
            index = rand
            return index
        breakPoint += 1
        if breakPoint >= 100:
            return 0


# User Input
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
st.header("Input")
selected_label = st.selectbox("Select Digit [0-9]", [i for i in range(10)])
index = getImage(selected_label, Y_test)

if st.button("Random"):
    index = getImage(random.randint(0, 9), Y_test)

fig_input, ax_input = plt.subplots()
fig_input.suptitle(f"Actual Label: {Y_test[index]}")
ax_input.imshow(X_test[index], cmap="gray")

st.pyplot(fig_input)

X_train_re = X_train.reshape((60000, 28 * 28))
X_test_re = X_test.reshape((10000, 28 * 28))

Y_train_re = to_categorical(Y_train)
Y_test_re = to_categorical(Y_test)

print(model.evaluate(X_test_re, Y_test_re))
input_img = X_test_re[index].reshape(1, 784)


st.header("Prediction")
y_pred = model.predict(input_img)

st.write("Prediction Results (Probability of being each label)")
predict_result = []
labels = [i for i, pred in enumerate(y_pred[0])]

for i, pred in enumerate(y_pred[0]):
    # st.write(f"Label {i}: {pred:.4f}") # Show probability of prediction being belong to what class (0-9)
    predict_result.append(round(pred * 100, 3))

results = pd.DataFrame(predict_result, index=labels, columns=["Percentage"])
results.index.name = "Label"
st.write(results)

loss = model.evaluate(X_test_re, Y_test_re)[0]
accuracy = round(model.evaluate(X_test_re, Y_test_re)[1] * 100, 3)

st.write(f"Loss: {loss}")
st.write(f"Accuracy: {accuracy}%")


# Plot the label
fig_res, ax_res = plt.subplots()
fig_res.suptitle(
    f"Actual Label: {Y_test[index]} & Predicted Label: {y_pred.argmax(axis=1)}"
)
ax_res.set_xlabel(
    f"Correct: {Y_test[index] == y_pred.argmax(axis=1).tolist()[0]} ({Y_test[index]}, {y_pred.argmax(axis=1).tolist()[0]})"
)
ax_res.imshow(X_test[index], cmap="gray")
st.pyplot(fig_res)

st.divider()
st.header("Model Performance Overview")
# st.subheader(f"Loss: {model.evaluate(X_test_re, Y_test_re)[0]}")
# st.subheader(f"Accuracy: {model.evaluate(X_test_re, Y_test_re)[1]}")

y_true = Y_test

y_pred = model.predict(X_test_re)
y_pred = y_pred.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
dist = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[i for i in range(10)]
)

fig, ax = plt.subplots()
fig.suptitle("Confusion Matrix")
dist.plot(ax=ax)
st.pyplot(fig)

# Load pickle history file
# url = "https://github.com/Purinat33/streamlit_stock_price/raw/master/history.pkl"
# file_path = "history.pkl"
# urllib.request.urlretrieve(url, file_path)

# # Load the saved history
# with open(file_path, "rb") as file:
#     history_dict = pickle.load(file)

# model_history = pd.DataFrame(history_dict)
