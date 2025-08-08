import streamlit as st
from ImageModel import promptgen, text2image
from io import BytesIO
import os
import time
from typing import Literal
from PIL import Image
import tempfile

def app():
    # Set the title of the web page
    st.title("Audio2Art: Transforming Audio Prompts into Visual Creations")
    
    # Create a file uploader for wav files
    upload_file = st.file_uploader("Choose your .wav audio file", type=["wav"])
    
    # Dropdown for selecting the model option
    option = st.selectbox(
        'Select Model',
        ("runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1")
    )
    
    # Create session state to store results
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None
        st.session_state.image_buffer = None
    
    # Create a form with a submit button
    with st.form("my_form"):
        submit = st.form_submit_button(label="Submit Audio File!")
    
    # Process when submit is pressed (outside the form)
    if submit:
        if upload_file is not None:
            # Show a spinner while processing
            with st.spinner("Generating Image ... It may take some time."):
                try:
                    # Save uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(upload_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Generate prompt from audio using the file path
                    prompt = promptgen(tmp_path)
                    st.write(f"Generated prompt: {prompt}")
                    
                    # Clean up the temporary file
                    os.unlink(tmp_path)
                    
                    # Generate image from prompt
                    im, start, end = text2image(prompt, option)
                    
                    # Create a buffer for the image and store in session state
                    buf = BytesIO()
                    im.save(buf, format="PNG")
                    st.session_state.generated_image = im
                    st.session_state.image_buffer = buf
                    
                    # Calculate processing time
                    processing_time = end - start
                    
                    # Display success message with processing time
                    st.success(f"Image generated in {processing_time:.2f} seconds!")
                    
                    # Display the generated image
                    st.image(im)
                    
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
        else:
            st.error("Please upload an audio file!")
    
    # Download button outside the form
    if st.session_state.generated_image is not None:
        st.download_button(
            label="Download Image",
            data=st.session_state.image_buffer.getvalue(),
            file_name="generated_image.png",
            mime="image/png"
        )
    
    # Add sidebar with guide and examples
    with st.sidebar:
        st.header("How to use Audio2Art")
        st.write("""
        1. Upload a .wav audio file containing your voice prompt
        2. Select the model you want to use
        3. Click 'Submit Audio File!'
        4. Wait for the image to be generated
        5. Download the generated image if desired
        """)
        
        st.header("Examples")
        st.write("""
        Try these voice prompts:
        - "A beautiful sunset over mountains with reflections in a lake"
        - "A futuristic city with flying cars and tall skyscrapers"
        - "A cute puppy playing in a garden full of flowers"
        """)

# Run the app when the script is executed directly
if __name__ == "__main__":
    app() 