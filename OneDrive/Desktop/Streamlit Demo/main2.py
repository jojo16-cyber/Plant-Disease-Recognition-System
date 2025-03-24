import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the cleaned dataset
df = pd.read_csv("crops_data_cleaned.csv")

# Rename columns for easy access
df = df.rename(columns={
    "Estimated Cost per Acre (₹)": "Cost_per_Acre",
    "Common Diseases": "Disease",
    "Water Requirement per Acre (Liters/Season)": "Water_Requirement",
    "Estimated Profit per Acre (₹)": "Profit_per_Acre"
})

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

# Encode categorical variables using separate LabelEncoders
month_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

# Ensure clean text values
df["Month"] = df["Month"].astype(str).str.strip()
df["Crop"] = df["Crop"].astype(str).str.strip()

# Fit encoders
df["Month_encoded"] = month_encoder.fit_transform(df["Month"])
df["Crop_encoded"] = crop_encoder.fit_transform(df["Crop"])

# Train the ML model
X = df[["Month_encoded"]]
y = df["Crop_encoded"]
model = RandomForestClassifier()
model.fit(X, y)

# Reverse mapping for crops
crop_mapping = dict(zip(df["Crop_encoded"], df["Crop"]))

# Streamlit UI
st.title("🌾 Crop Selection & Recommendation")

# Step 1: Select Month
selected_month = st.selectbox("📅 Select a Month:", sorted(df["Month"].unique()))

if selected_month in df["Month"].values:
    # Step 2: Display all crops available in the selected month
    crops_in_month = df[df["Month"] == selected_month]

    st.write(f"### 🌱 Crops Available in {selected_month}:")
    st.table(crops_in_month[["Crop", "Cost_per_Acre", "Disease", "Water_Requirement", "Profit_per_Acre"]])

    # ML-Based Crop Recommendation
    if st.button("🔍 Recommend the Best Crop"):
        if selected_month not in month_encoder.classes_:
            st.error("⚠️ Selected month is not in the training data. Please choose another.")
            st.stop()  # Stop execution to prevent further errors

        month_encoded = month_encoder.transform([selected_month])[0]
        recommended_crop_encoded = model.predict([[month_encoded]])[0]
        recommended_crop = crop_mapping[recommended_crop_encoded]

        st.success(f"🌟 Recommended Crop for {selected_month}: **{recommended_crop}**")
else:
    st.error("⚠️ Selected month is not in the training data. Please choose another.")
