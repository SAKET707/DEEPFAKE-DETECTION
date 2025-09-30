import streamlit as st
from prediction_helper import predict 

st.title("DEEPFAKE DETECTION")
st.warning("⚠️ Please ensure you have a **stable internet connection** for smooth image upload and prediction.")

uploaded_file = st.file_uploader("UPLOAD THE FILE",type=['jpg','png'])

if uploaded_file:
    image_path = 'temp_file.jpg'
    with open(image_path,'wb') as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file,caption='UPLOADED FILE',use_container_width=True)
        prediction = predict(image_path)
        st.info(f"PREDICTED CLASS : {prediction}")