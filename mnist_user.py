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
import time  # For measuring execution time

# TODO: Custom user's image input (file upload is the goal for now)
# 7th July 2023
from keras.utils import load_img, img_to_array

# In the future we might do CNN entirely and not have to convert images like this

# import os
# st.write('DEV')
st.title("MNIST Handwritten Digit Prediction")
st.write("Now you can run MNIST yourself!")
st.divider()

# IMPORTING THE MODEL
# Path starts at ~/users/user_name for some reasom
# Update: File path for opening in Streamlit and for running in normal python are DIFFERENT

# Load locally
# filepath = os.path.abspath("./Desktop/youtube/streamlit/my_mnist.h5")
# model = keras.models.load_model(filepath)


# Moved to functions to save space
# https://docs.streamlit.io/library/advanced-features/caching#basic-usage
@st.cache_data
def load_img(url, filename):
    urllib.request.urlretrieve(url, filename)
    image = Image.open(filename)
    return image


@st.cache_resource
def load_model(url, filename):
    file_path = keras.utils.get_file(filename, origin=url)
    model = keras.models.load_model(file_path)
    return model


@st.cache_data
def get_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    dist = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[i for i in range(10)]
    )

    fig, ax = plt.subplots()
    fig.suptitle("Confusion Matrix")
    dist.plot(ax=ax)

    return fig


# No need to do caching here
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


# Load mnist function
# The reason we made a separate function is to cache it
# It is small, but barely faster is faster anyway
@st.cache_data
def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    return (X_train, Y_train), (X_test, Y_test)


@st.cache_data
def reshapeX(X, first_dim, second_dim, third_dim):
    X_re = X.reshape((first_dim, second_dim * third_dim))
    return X_re


@st.cache_data
def catalogueY(Y):
    Y_re = to_categorical(Y)
    return Y_re


# We already cache_resource the model so this one with variables type of numpys
# We can use cache_data
@st.cache_data
def evaluate_model(_model, X_test_re, Y_test_re):
    loss = _model.evaluate(X_test_re, Y_test_re)[0]
    accuracy = round(_model.evaluate(X_test_re, Y_test_re)[1] * 100, 3)

    return loss, accuracy


def main():
    # Load from GitHub
    # Note that the file URL needs to be for the `raw` file
    url = "https://github.com/Purinat33/Streamlit-MNIST/raw/master/my_mnist.h5"
    model = load_model(url, "my_mnist.h5")

    # Load image
    img_url = (
        "https://github.com/Purinat33/Streamlit-MNIST/raw/master/mnist_overview_95.png"
    )
    st.header("Model Architectural Overview")
    img = load_img(img_url, "mnist_overview_95.png")
    st.image(img)

    # Dataset
    (X_train, Y_train), (X_test, Y_test) = load_mnist()

    # User Input Section
    st.header("Input")
    selected_label = st.selectbox("Select Digit [0-9]", [i for i in range(10)])
    index = getImage(selected_label, Y_test)

    if st.button("Random"):
        index = getImage(random.randint(0, 9), Y_test)

    fig_input, ax_input = plt.subplots()
    fig_input.suptitle(f"Actual Label: {Y_test[index]}")
    ax_input.imshow(X_test[index], cmap="gray")

    st.pyplot(fig_input)

    # Flatten the input X
    # X_train_re = X_train.reshape((60000, 28 * 28))
    X_train_re = reshapeX(X_train, 60000, 28, 28)
    # X_test_re = X_test.reshape((10000, 28 * 28))
    X_test_re = reshapeX(X_test, 10000, 28, 28)

    # Labeling the Y
    # Y_train_re = to_categorical(Y_train)
    Y_train_re = catalogueY(Y_train)
    # Y_test_re = to_categorical(Y_test)
    Y_test_re = catalogueY(Y_test)

    # print(model.evaluate(X_test_re, Y_test_re))
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

    # loss = model.evaluate(X_test_re, Y_test_re)[0]
    # accuracy = round(model.evaluate(X_test_re, Y_test_re)[1] * 100, 3)
    loss, accuracy = evaluate_model(model, X_test_re, Y_test_re)
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
    st.header("Upload your own image!")

    # A part for the user to upload their own handwriting image
    img_upload = st.file_uploader(
        "Upload your own handwriting image", type=["jpg", "png"]
    )
    if img_upload is not None:
        user_image = Image.open(img_upload)
        st.image(user_image)

    st.header("Model Performance Overview")
    # st.subheader(f"Loss: {model.evaluate(X_test_re, Y_test_re)[0]}")
    # st.subheader(f"Accuracy: {model.evaluate(X_test_re, Y_test_re)[1]}")

    y_true = Y_test

    y_pred = model.predict(X_test_re)
    y_pred = y_pred.argmax(axis=1)

    cm_fig = get_confusion_matrix(y_true, y_pred)
    st.pyplot(cm_fig)

    st.divider()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    execution_time = end_time - start_time
    st.write(f"App completed running in {round(execution_time, 3)} seconds")
