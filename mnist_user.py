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

# Scikit-image for reszing array
# from skimage.transform import resize

# TODO: Canvas for users to draw directly
# streamlit_drawable_canvas can be used to create a drawing board for a user to work on
# GitHub: andfanilo/streamlit-drawable-canvas
# from streamlit_drawable_canvas import st_canvas

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


# def createCanvas(stroke_color, stroke_width=4):
#     canvas = st_canvas(
#         fill_color="rgba(0, 0, 0, 1.0)",
#         stroke_width=stroke_width,
#         stroke_color=stroke_color,
#         background_color="black",
#         update_streamlit=True,
#         height=560,
#         width=560,
#         drawing_mode="freedraw",
#         point_display_radius=0,
#         key="morb",
#     )
#     return canvas


def main():
    # Load from GitHub
    # Note that the file URL needs to be for the `raw` file
    url = "https://github.com/Purinat33/Streamlit-MNIST/raw/master/my_mnist.h5"
    model = load_model(url, "my_mnist.h5")
    st.sidebar.title("MNIST Playground")

    web_mode = st.sidebar.selectbox("Select Mode", ["Playground", "Demo"])
    st.sidebar.write(
        """
                     The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.
                     """
    )

    if web_mode == "Playground":
        # User upload/draw
        st.header("Upload your own image!")
        processed_imaged_url = (
            "https://github.com/Purinat33/Streamlit-MNIST/raw/master/image_process.png"
        )
        example_image = load_img(processed_imaged_url, "image_process.png")
        st.write("Example of what will happen to your uploaded file.")
        st.image(example_image)

        # A part for the user to upload their own handwriting image
        st.subheader("Upload Your Digit")
        img_upload = st.file_uploader(
            "Upload image here (No data shall be collected): ",
            type=["jpg", "png", "jpeg", "svg", "bmp"],
        )

        if img_upload is not None:
            # Preprocessed the data like the example
            user_image = Image.open(img_upload)
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

            # Hide the ticks
            for axis in ax.flat:
                axis.set_xticks([])
                axis.set_yticks([])

            # Original Image
            ax[0, 0].set_title("Original Image")
            ax[0, 0].imshow(user_image)

            # Resized Image (28 * 28)
            resized_image = user_image.resize((28, 28))
            ax[0, 1].set_title("Resized Image")
            ax[0, 1].imshow(resized_image)

            # Grayscale Image
            grayscaled_image = resized_image.convert("L")
            ax[0, 2].set_title("Grayscaled Image")
            ax[0, 2].imshow(grayscaled_image, cmap="gray")

            # Invert the image (so the text is white, and background is black)
            inverted_image = 255 - np.array(grayscaled_image)
            ax[1, 0].set_title("Inverted Image")
            ax[1, 0].imshow(inverted_image, cmap="gray")

            # Normalize the image (divided by 255 to make them between 0 - 1)
            normalized_image = inverted_image / 255.0
            ax[1, 1].set_title("Normalized Image")
            ax[1, 1].imshow(normalized_image, cmap="gray")

            # Reshaped image
            reshaped_image = normalized_image.reshape((28, 28))
            ax[1, 2].set_title("Reshaped Image")
            ax[1, 2].imshow(reshaped_image, cmap="gray")

            st.pyplot(fig)

            # The actual data we're going to use
            reshaped_image = normalized_image.reshape((1, 28 * 28))

            # true_label = st.selectbox(
            #     "What digit does your image represent?", [i for i in range(10)]
            # )
            predicted_label = model.predict(reshaped_image)
            predicted_label = predicted_label.argmax(axis=1)

            # st.subheader(f"True Value: {true_label}")
            st.subheader(f"Predicted Value: {predicted_label.tolist()[0]}")
            st.divider()

            # if true_label == predicted_label.tolist()[0]:
            #     st.subheader(":green[CORRECT]")
            # else:
            #     st.subheader(":red[INCORRECT]")

            # st.header("Draw your digit")
            # # Streamlit_canvas
            # # Draw the canvas

            # # Eraser = white pen
            # # Pen = black pen
            # st.write("The Canvas is yours to play!")
            # mode = st.selectbox("Pen/Eraser", ["Pen", "Eraser"])
            # stroke_color = "white"
            # stroke_width = 4
            # if mode == "Pen":
            #     stroke_color = "white"
            #     stroke_width = 4
            # else:
            #     stroke_color = "black"
            #     stroke_width = 20

            # canvas = createCanvas(stroke_color, stroke_width)
            # if canvas.image_data is not None:
            #     drawn_image = canvas.image_data  # Numpy array btw
            #     # We will do prediction here
            #     # Data is in (560, 560, 4) shape
            #     # We have to resize to 1, 784 for prediction
            #     normalized_data = drawn_image / 255.0
            #     # Use skimage.transform.resize for better resizing than numpy resizing
            #     # resized_data = resize(normalized_data, (28, 28), anti_aliasing=True)

            #     # The data is still in (28, 28, 4) format and so we have to reshape the 2nd axis
            #     # resized_data = np.mean(resized_data, axis=2)
            #     flatten_resized_data = np.resize(normalized_data, (1, 784))
            #     # Prediction
            #     # flatten_resized_data = resized_data.reshape((1, 28 * 28))
            #     st.write(flatten_resized_data)
            #     st.subheader("Prediction")
            #     drawn_prediction = model.predict(flatten_resized_data)
            #     y_drawn = drawn_prediction.argmax(axis=1)

            # st.write(f"Label: {y_drawn}")
        st.divider()

    else:
        # Load image
        img_url = "https://github.com/Purinat33/Streamlit-MNIST/raw/master/mnist_overview_95.png"
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
