import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyBdJucnG3ujeokv7OzyPT9pRkuSEzs4PQ0")
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Load Fertilizer and Fungicide Data
try:
    fertilizer_df = pd.read_csv("fertilizers.csv")  # Ensure this file is in the same directory
except FileNotFoundError:
    st.error("fertilizers.csv file not found. Please make sure it's in the same directory.")
    fertilizer_df = pd.DataFrame(columns=["Crop Disease", "Fertilizer", "Fungicide / Insecticide", "Estimated Cost -1", "Estimated Cost -2"])

# Function to get response from Gemini
def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)  # Using the model instance directly
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini API: {str(e)}")
        return get_fallback_response(prompt)

# Fallback response function when API fails
def get_fallback_response(prompt):
    # Extract disease name from prompt
    disease_name = ""
    if "plant disease:" in prompt:
        disease_name = prompt.split("plant disease:")[1].split(".")[0].strip()
    elif "has identified:" in prompt:
        disease_name = prompt.split("has identified:")[1].split(".")[0].strip()
    
    return f"""
    # {disease_name} - Disease Information

    ## About the Disease
    {disease_name} is a plant disease that can affect crop health and yield. Without connectivity to the Gemini API, I'm providing general guidance.

    ## General Recommendations:
    1. For bacterial diseases: Consider copper-based fungicides
    2. For fungal diseases: Consider sulfur-based fungicides
    3. For viral diseases: Focus on prevention through clean tools and removing infected plants
    
    ## Preventive Measures:
    - Practice crop rotation
    - Maintain proper spacing between plants
    - Ensure good air circulation
    - Use drip irrigation instead of overhead watering
    - Remove and destroy infected plant material
    
    ## Consult a Local Extension Office:
    For specific treatment recommendations, please consult your local agricultural extension office.
    """

# Tensorflow Model Prediction with error handling
def model_prediction(test_image):
    try:
        model_path = "trained_model.keras"
        
        # Check if model exists
        if not os.path.exists(model_path):
            st.error(f"Model file {model_path} not found. Using demo mode.")
            # For demo purposes, return a fixed index
            return 16  # Peach Bacterial spot
        
        model = tf.keras.models.load_model(model_path)
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        
        with st.spinner("Analyzing image..."):
            prediction = model.predict(input_arr)
            result_index = np.argmax(prediction)
            return result_index
            
    except Exception as e:
        st.error(f"Error in model prediction: {str(e)}")
        # For demo purposes, return a fixed index 
        return 16  # Peach Bacterial spot

# Main UI
st.header("Plant Disease Recognition System")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #141F32;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a notice about demo mode
if not os.path.exists("trained_model.keras"):
    st.warning("⚠️ Running in demo mode: The trained model file is missing. Predictions will be simulated.")

test_image = st.file_uploader("Choose an image")

col1, col2 = st.columns(2)
with col1:
    show_image = st.button("Show Image")
with col2:
    predict_button = st.button("Predict")

if test_image is not None and show_image:
    st.image(test_image, caption="Uploaded Image")

if test_image is not None and predict_button:
    with st.spinner("Analyzing image..."):
        result_index = model_prediction(test_image)
    
    # Define class names
    class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
    
    detected_disease = class_name[result_index]
    formatted_disease = detected_disease.replace('_', ' ')
    st.success(f"🌿 Detected Disease: **{formatted_disease}**")

    import difflib

    # Function to find the best match from CSV
    def find_best_match(disease_name, disease_list):
        matches = difflib.get_close_matches(disease_name, disease_list, n=1, cutoff=0.5)
        return matches[0] if matches else None

    # Get all disease names from CSV
    csv_diseases = fertilizer_df["Crop Disease"].tolist()

    # Find the best match for detected disease
    best_match = find_best_match(formatted_disease, csv_diseases)

    if best_match:
        match = fertilizer_df[fertilizer_df["Crop Disease"] == best_match]
        st.success(f"🛠 **Recommended Treatment for {best_match}**")
        
        for index, row in match.iterrows():
            st.write(f"**🌾 Crop:** {row['Crop Disease']}")
            st.write(f"🔹 **Recommended Fertilizer:** {row['Fertilizer']}")
            st.write(f"🦠 **Recommended Fungicide/Insecticide:** {row['Fungicide / Insecticide']}")
            st.write(f"💰 **Estimated Cost:** {row['Estimated Cost -1']} | {row['Estimated Cost -2']}")
            
            # Get enhanced explanation from Gemini
            prompt = f"""
            You are a helpful agricultural expert. 
            Please provide a detailed explanation about the plant disease: {formatted_disease}.
            Include information about:
            1. What causes this disease
            2. How it affects the plant
            3. How to properly apply the recommended treatment: {row['Fertilizer']} and {row['Fungicide / Insecticide']}
            4. Preventive measures for future
            
            Respond in a friendly, helpful tone as if you're talking to a farmer who needs guidance.
            Keep the response concise but informative.
            """
            
            with st.spinner("Generating detailed explanation..."):
                gemini_response = get_gemini_response(prompt)
                st.subheader("🤖 Expert Explanation")
                st.write(gemini_response)
    else:
        st.warning("⚠ No specific treatment found in the dataset for this disease.")
        
        # Use Gemini to provide information when database doesn't have a match
        prompt = f"""
        You are a helpful agricultural expert. 
        The plant disease detection system has identified: {formatted_disease}.
        
        Please provide:
        1. A detailed explanation of this disease
        2. Recommended fertilizers and fungicides/insecticides that would be effective
        3. Approximate costs of these treatments
        4. Application methods and preventive measures
        
        Respond in a friendly, helpful tone as if you're talking to a farmer who needs guidance.
        Format your response with clear sections and be specific about treatment recommendations.
        200 words at max
        """
        
        with st.spinner("Searching for treatment information..."):
            gemini_response = get_gemini_response(prompt)
            st.subheader("Expert Recommendation")
            st.write(gemini_response)
