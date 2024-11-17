# app.py
import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import base64
from io import BytesIO
from PIL import Image

# Import GAN model classes
from gan_models import StyleGAN, CycleGAN, TextToImageGAN, SRGAN, MedicalGAN

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load GAN models (Assumes models are pre-trained and saved in the "models" folder)
stylegan = StyleGAN()
cyclegan = CycleGAN()
text_to_image_gan = TextToImageGAN()
srgan = SRGAN()
medical_gan = MedicalGAN()

# Data model for incoming requests
class GANRequest(BaseModel):
    gan_type: str
    input_data: str = None  # For text or base64 image data

# API endpoint to generate image
@app.post("/generate_image")
async def generate_image(request: GANRequest):
    try:
        # Generate image based on GAN type
        if request.gan_type == "stylegan":
            result = stylegan.generate_image()
        elif request.gan_type == "cyclegan":
            result = cyclegan.translate_image(request.input_data)
        elif request.gan_type == "text_to_image":
            result = text_to_image_gan.generate_from_text(request.input_data)
        elif request.gan_type == "srgan":
            result = srgan.enhance_resolution(request.input_data)
        elif request.gan_type == "medical_gan":
            result = medical_gan.synthesize_image()
        else:
            return {"error": "Unknown GAN type selected"}
        
        # Return generated image as base64-encoded string
        return {"image_data": result}
    except Exception as e:
        return {"error": str(e)}

# Streamlit frontend
def main():
    st.title("GAN Image Studio")

    # Dropdown for selecting GAN model
    gan_type = st.selectbox(
        "Select GAN Model",
        ("StyleGAN (Image Generation)", "CycleGAN (Image Translation)", 
         "Text-to-Image GAN", "SRGAN (Image Super-Resolution)", 
         "Medical Imaging GAN")
    )

    # Input area for user prompt
    if gan_type == "Text-to-Image GAN":
        text_prompt = st.text_input("Enter a description for the image")
        input_data = text_prompt
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        input_data = base64.b64encode(uploaded_file.read()).decode() if uploaded_file else None

    # Button to trigger image generation
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            # Send request to FastAPI backend
            response = requests.post(
                "http://localhost:8000/generate_image", 
                json={"gan_type": gan_type.lower().replace(" ", "_"), "input_data": input_data}
            )

            # Display generated image
            if response.status_code == 200 and "image_data" in response.json():
                image_data = base64.b64decode(response.json()["image_data"])
                st.image(Image.open(BytesIO(image_data)), caption="Generated Image")
            else:
                st.error("Image generation failed.")

if __name__ == "__main__":
    main()
    app.run(host='0.0.0.0', port=8000)