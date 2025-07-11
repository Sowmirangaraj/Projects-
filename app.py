import streamlit as st
import numpy as np
import json
import pickle
import cv2
import tempfile

# Load model only once
@st.cache_resource
def load_model():
    with open("knn_check_valve_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# App settings
st.set_page_config(page_title="Valve Defect Detection", layout="centered")
st.title("🔍 Valve Defect Detection from QR Code")

# Upload QR image
uploaded_file = st.file_uploader("📤 Upload QR Code Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded QR Code", use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    # Decode QR using OpenCV
    image = cv2.imread(img_path)
    qr = cv2.QRCodeDetector()
    data, _, _ = qr.detectAndDecode(image)

    if not data:
        st.error("❌ Could not detect or decode QR code.")
    else:
        try:
            # Convert QR string to dict
            measurement = json.loads(data)

            # Display extracted measurement
            st.subheader("🧾 Extracted Measurements:")
            st.json(measurement)

            # Prepare data for model
            features = ['valve_id', 'body_height', 'inlet_radius', 'outlet_radius', 'disc_thickness', 'spring_length']
            input_data = np.array([[float(measurement[feat]) for feat in features]])

            # Prediction
            prediction = model.predict(input_data)[0]

            # Show result
            st.subheader("🔎 Prediction Result:")
            if prediction == 0:
                st.error("❌ Defective")
            else:
                st.success("✅ Not Defective")

        except Exception as e:
            st.error(f"⚠️ Error processing QR data: {e}")
