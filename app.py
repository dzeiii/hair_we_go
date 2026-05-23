import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile

# Allow truncated image handling for user uploads
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Page Configuration Setup
st.set_page_config(page_title="AI Hairstyle Recommender", page_icon="✂️", layout="centered")

# 2. Load Your Trained Brain Safely
@st.cache_resource
def load_my_model():
    # Looks for the file in the exact same directory
    return tf.keras.models.load_model('hairstyle_model.h5')

try:
    model = load_my_model()
    CLASSES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
except Exception as e:
    st.error("Model file 'hairstyle_model.h5' not found in this folder. Please upload it to your folder environment.")
    st.stop()

# 3. User Interface Content
st.title("AI Hairstyle Recommender ✂️")
st.write("Upload a clear front-facing photo of yourself to find your face shape and the best matching hairstyles!")

uploaded_file = st.file_uploader("Choose a selfie photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded file
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Uploaded Selfie', use_container_width=True)
    
    st.info("🔄 Processing image and scanning facial structure...")
    
    # Preprocess to match exactly how the AI model was trained
    img = image.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run prediction logic
    prediction = model.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Show classification results
    st.success(f"### Target Match Identified: **{predicted_class}** ({confidence:.1f}% Confidence)")
    
    # 4. Recommendation Mapping Rules logic
    st.write("---")
    st.write("### 💇‍♂️ Recommended Hairstyles for Your Face Shape:")
    
    if predicted_class == 'Heart':
        st.markdown("- **Side-Swept Fringes:** Softens a wide forehead.\n- **Long Layers:** Adds volume around the lower jawline.\n- **Textured Pixie Cut:** Enhances natural bone structure features.")
    elif predicted_class == 'Oblong':
        st.markdown("- **Side Partings:** Visually widens the facial structure look.\n- **Fringe/Bangs:** Cuts down on vertical length perception.\n- **Voluminous Curls:** Adds balancing width to side profiles.")
    elif predicted_class == 'Oval':
        st.markdown("- **Blunt Bob:** Complements balanced structural symmetries.\n- **Slicked Back Undercut:** Clean style showcasing smooth contours.\n- **Long Waves:** Universally framing for oval configurations.")
    elif predicted_class == 'Round':
        st.markdown("- **Pompadour:** Adds flattering height to vertical features.\n- **Asymmetrical Cuts:** Breaks up rounder circular symmetries.\n- **Long Layered Shag:** Creates structural angles along cheeks.")
    elif predicted_class == 'Square':
        st.markdown("- **Soft Layered Waves:** Tones down sharp, boxy angles.\n- **Side Swept Bob:** Shifts visual focus away from jawlines.\n- **Buzz Cut:** Accents strong, athletic structural lines.")
