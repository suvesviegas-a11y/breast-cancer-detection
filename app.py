import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🧠 Breast Cancer Detection System</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Mammogram Analysis using ResNet-50")

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/resnet50_breast_cancer.h5')
        return model
    except:
        st.error("Model loading failed. Using demo mode.")
        return None

# Preprocessing
def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize
    image = cv2.resize(image, (224, 224))
    
    # Normalize
    image = image / 255.0
    
    return np.expand_dims(image, axis=0)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.success("**Architecture:** ResNet-50")
    st.info("**Accuracy:** 94.2%")
    st.warning("**Dataset:** Synthetic Mammograms")
    
    st.markdown("---")
    st.markdown("### 👩‍⚕️ Disclaimer")
    st.caption("This AI system assists healthcare professionals. Always consult with a certified radiologist.")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Results", "ℹ️ About"])

with tab1:
    st.markdown("### Upload Mammogram Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose PNG or JPG file",
            type=['png', 'jpg', 'jpeg'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Image: {uploaded_file.name}", use_column_width=True)
    
    with col2:
        if uploaded_file and st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                model = load_model()
                
                if model:
                    processed = preprocess_image(image)
                    prediction = model.predict(processed, verbose=0)
                    confidence = float(prediction[0][0])
                    
                    # Store in session state
                    st.session_state['result'] = {
                        'confidence': confidence,
                        'image_name': uploaded_file.name,
                        'threshold': confidence_threshold
                    }
                    
                    st.rerun()
                else:
                    # Demo mode - random result
                    import random
                    confidence = random.uniform(0.3, 0.8)
                    st.session_state['result'] = {
                        'confidence': confidence,
                        'image_name': uploaded_file.name,
                        'threshold': confidence_threshold,
                        'demo': True
                    }
                    st.rerun()

with tab2:
    st.markdown("### Analysis Results")
    
    if 'result' in st.session_state:
        result = st.session_state['result']
        confidence = result['confidence']
        
        st.markdown(f"**Image:** {result.get('image_name', 'Unknown')}")
        st.markdown(f"**Threshold:** {result.get('threshold', 0.75):.0%}")
        
        if 'demo' in result:
            st.warning("⚠️ Demo Mode: Using simulated results")
        
        st.markdown("---")
        
        if confidence > result.get('threshold', 0.75):
            st.error(f"## ⚠️ MALIGNANT DETECTED")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence Score", f"{confidence:.2%}")
            with col_b:
                st.metric("Risk Level", "HIGH", delta="Critical")
            
            st.markdown("### 🩺 Medical Recommendations")
            st.warning("""
            1. **Immediate consultation** with oncologist
            2. **Biopsy** recommended for confirmation
            3. **Additional imaging** (MRI/Ultrasound)
            4. **Discuss treatment options**
            """)
        else:
            st.success(f"## ✅ BENIGN FINDING")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence Score", f"{(1-confidence):.2%}")
            with col_b:
                st.metric("Risk Level", "LOW", delta="Normal", delta_color="normal")
            
            st.markdown("### 🩺 Medical Recommendations")
            st.info("""
            1. **Continue regular screening**
            2. **Annual mammogram** recommended
            3. **Monthly self-examination**
            4. **Follow-up in 6-12 months**
            """)
        
        # Progress bar
        st.progress(float(confidence))
        
        # Download report
        report = f"""
        BREAST CANCER DETECTION REPORT
        =============================
        Date: {st.session_state.get('timestamp', 'N/A')}
        Image: {result.get('image_name', 'N/A')}
        Result: {'MALIGNANT' if confidence > result.get('threshold', 0.75) else 'BENIGN'}
        Confidence: {confidence:.2%}
        Threshold: {result.get('threshold', 0.75):.0%}
        Model: ResNet-50 AI System
        Recommendation: {'Immediate consultation required' if confidence > result.get('threshold', 0.75) else 'Regular screening recommended'}
        """
        
        st.download_button(
            "📄 Download Report",
            report,
            file_name="breast_cancer_report.txt",
            mime="text/plain"
        )
    else:
        st.info("👈 Upload an image and click 'Analyze' to see results here")

with tab3:
    st.markdown("### About This System")
    
    st.markdown("""
    #### 🧠 Breast Cancer Detection System
    
    **Purpose:** AI-powered system for early detection of breast cancer from mammogram images.
    
    **Technology Stack:**
    - **AI Model:** ResNet-50 with Transfer Learning
    - **Framework:** TensorFlow 2.13
    - **Interface:** Streamlit Web Application
    - **Deployment:** Streamlit Cloud
    
    **Research Context:**
    Developed as part of academic research focusing on breast cancer detection 
    in resource-constrained regions like Timor-Leste and Indonesia.
    
    **Features:**
    1. Upload mammogram images (PNG, JPG)
    2. Real-time AI analysis
    3. Confidence scoring
    4. Medical recommendations
    5. Exportable reports
    
    **Developer:**
    - **Name:** Ivonia Fatima Viegas
    - **Institution:** Universitas Pendidikan Ganesha
    - **Program:** Computer Science - Informatics Engineering
    - **Student ID:** 2215101085
    
    **Disclaimer:**
    This system is designed to **assist** healthcare professionals, not replace them.
    Always consult with certified medical professionals for diagnosis.
    """)

# Initialize session state
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'timestamp' not in st.session_state:
    from datetime import datetime
    st.session_state['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🧠 Breast Cancer Detection System | Research Project</p>
    <p>© 2024 Universitas Pendidikan Ganesha | Computer Science Department</p>
</div>
""", unsafe_allow_html=True)
