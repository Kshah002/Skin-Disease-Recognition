import base64
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Define the background image URL
background_image_path = "WebappImage.jpeg"
# Replace with the URL of your background image

# Add custom CSS to style the background
custom_css = f"""
    <style>
        body {{
            background-image: url("data:image/jpeg;base64,{base64.b64encode(open(background_image_path, "rb").read()).decode()}");
            background-size: cover;
        }}
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)
st.header("Skin Diseases Prediction")


def main():
    file_uploaded = st.file_uploader("Choose the file", type=["jpg", "png", "jpeg"])
    if file_uploaded  is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)


def predict_class(image):
    # Load the pre-trained model from TensorFlow Hub
    loaded_model_url = "my_Xce_TF_model"
    shape = (224, 224, 3)
    model = tf.keras.Sequential([hub.KerasLayer(loaded_model_url, input_shape=shape)])

    # Resize and preprocess the image
    test_image = image.resize((224, 224))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    class_names = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal']

    # Predict the class of the uploaded image
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]

    result = "The image uploaded is: {}".format(image_class)
    return result


if __name__ == "__main__":
    main()

# press to stop the Streamlit app
if st.button("Exit"):
    st.stop()
