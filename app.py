import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the trained model
model = tf.keras.models.load_model('plant_identification_model2.h5')

# Create label mapping based on subdirectory names
main_data_dir = 'Segmented Medicinal Leaf Images'
label_mapping = {i: label for i, label in enumerate(sorted(os.listdir(main_data_dir)))}

# Streamlit UI
st.title("Ayurvedic Plant Species Identification")

# Upload an image for prediction
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Check if the uploaded file is an image
    if uploaded_image.type.startswith('image/'):
        # Display the uploaded image
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict the image
        def preprocess_image(image):
            image = load_img(image, target_size=(224, 224))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            preprocessed_image = preprocess_input(image_array)
            return preprocessed_image

        if st.button("Predict"):
            preprocessed_image = preprocess_image(uploaded_image)
            predictions = model.predict(preprocessed_image)

            # Map model's numeric predictions to labels
            predicted_label_index = np.argmax(predictions)
            predicted_label = label_mapping[predicted_label_index]

            # Calculate accuracy (confidence)
            confidence = np.max(predictions)  # The highest predicted probability

            # Check if the predicted label is among the valid plant labels
            valid_plant_labels = label_mapping.values()
            if predicted_label in valid_plant_labels:
                # Display prediction and accuracy
                st.write(f"Predicted Label: {predicted_label}")
                st.write(f"Accuracy: {confidence * 100:.2f}%")
            else:
                st.error("Invalid Image. Please upload an image of a plant or leaf.")
    else:
        st.error("Please upload a valid image file (JPEG or PNG).")

# Display some sample images from your dataset
st.header("Sample Plant Images")
class_folders = os.listdir(main_data_dir)
num_samples = min(len(class_folders), 5)  # Show up to 5 samples
images_per_row = 5

for i in range(num_samples):
    class_folder = class_folders[i]
    class_folder_path = os.path.join(main_data_dir, class_folder)
    image_files = [f for f in os.listdir(class_folder_path) if f.endswith('.jpg')]

    if image_files:
        st.subheader(class_folder)
        fig, axs = plt.subplots(1, images_per_row, figsize=(15, 15))

        for j in range(images_per_row):
            image_path = os.path.join(class_folder_path, image_files[j])
            img = mpimg.imread(image_path)
            axs[j].imshow(img)
            axs[j].set_title(f"Sample {j + 1}")
            axs[j].axis('off')

        st.pyplot(fig)

# Add any additional content or information about your model or dataset here.
