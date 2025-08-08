import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from diffusers import StableDiffusionPipeline
import soundfile as sf
import librosa as lb
import numpy as np
import time
from typing import Literal, Tuple
from PIL import Image

def promptgen(file):
    """
    Process an audio file and generate a transcription of its content.
    
    Parameters:
    file (str): Path to the audio file
    
    Returns:
    str: Transcribed text from the audio file
    """
    try:
        tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')
        model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
        
        print(f"Loading audio file: {file}")
        # Try to load the audio file with error handling
        try:
            waveform, rate = lb.load(file, sr=16000, mono=True)
        except Exception as e:
            print(f"Error loading audio with librosa: {e}")
            # Try alternative loading with soundfile directly
            import soundfile as sf
            audio_data, rate = sf.read(file)
            if len(audio_data.shape) > 1:  # Convert stereo to mono if needed
                audio_data = audio_data.mean(axis=1)
            waveform = lb.resample(audio_data, orig_sr=rate, target_sr=16000)
        
        # Normalize the waveform
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)
        
        # Process with Wav2Vec
        input_values = tokenizer(waveform, return_tensors='pt').input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        
        transcription = tokenizer.batch_decode(predicted_ids)
        
        # If transcription is empty or just whitespace, return a default message
        if not transcription[0].strip():
            return "a beautiful abstract painting with vibrant colors"
        
        return transcription[0]
    except Exception as e:
        print(f"Error in promptgen: {e}")
        # Return a default prompt in case of any error
        return "a beautiful abstract painting with vibrant colors"

def text2image(prompt: str, repo_id: Literal["runwayml/stable-diffusion-v1-5", 
                                            "CompVis/stable-diffusion-v1-4",
                                            "stabilityai/stable-diffusion-2-1"]) -> Tuple[Image.Image, float, float]:
    """
    Generate an image from a text prompt using Stable Diffusion models.
    
    Parameters:
    prompt (str): Text description to generate the image from
    repo_id (Literal): Repository ID of the Stable Diffusion model to use
    
    Returns:
    tuple: (generated image, start time, end time)
    """
    # Set seed for reproducibility
    seed = 2024
    generator = torch.manual_seed(seed)
    
    # Set constants for the image generation process
    NUM_ITERS_TO_RUN = 1
    NUM_INFERENCE_STEPS = 50
    NUM_IMAGES_PER_PROMPT = 1
    
    # Record start time
    start = time.time()
    
    # Initialize the pipeline based on GPU availability
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            generator=generator
        )
        pipe = pipe.to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float32,
            generator=generator
        )
    
    # Generate images
    images = []
    for _ in range(NUM_ITERS_TO_RUN):
        result = pipe(
            prompt=prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
            generator=generator
        )
        images.extend(result.images)
    
    # Record end time
    end = time.time()
    
    # Return the first generated image and timing information
    return images[0], start, end

class ImageModel:
    def __init__(self):
        """
        Initialize the speech-to-text and text-to-image models
        """
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize Speech-to-Text model (Wav2Vec2)
        self.speech_model_name = "facebook/wav2vec2-base-960h"
        self.speech_tokenizer = None
        self.speech_model = None
        
        # Initialize Text-to-Image model (Stable Diffusion)
        self.image_model_name = "runwayml/stable-diffusion-v1-5"
        self.image_generator = None
        
        # Load models
        self.load_speech_model()
        self.load_image_model()
        
    def load_speech_model(self):
        """
        Load the speech-to-text model (Wav2Vec2)
        """
        print("Loading speech-to-text model...")
        try:
            self.speech_tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.speech_model_name)
            self.speech_model = Wav2Vec2ForCTC.from_pretrained(self.speech_model_name).to(self.device)
            print("Speech-to-text model loaded successfully.")
        except Exception as e:
            print(f"Error loading speech-to-text model: {e}")
    
    def load_image_model(self):
        """
        Load the text-to-image model (Stable Diffusion)
        """
        print("Loading text-to-image model...")
        try:
            self.image_generator = StableDiffusionPipeline.from_pretrained(
                self.image_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.image_generator = self.image_generator.to(self.device)
            print("Text-to-image model loaded successfully.")
        except Exception as e:
            print(f"Error loading text-to-image model: {e}")
    
    def speech_to_text(self, audio_file_path):
        """
        Convert speech to text using Wav2Vec2 model
        
        Parameters:
        audio_file_path (str): Path to the audio file
        
        Returns:
        str: Transcribed text
        """
        try:
            # Load audio file
            audio, sample_rate = sf.read(audio_file_path)
            
            # Resample if necessary
            if sample_rate != 16000:
                audio = lb.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Process audio with Wav2Vec2
            input_values = self.speech_tokenizer(
                audio, 
                return_tensors="pt",
                padding=True
            ).input_values.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                logits = self.speech_model(input_values).logits
            
            # Decode the prediction
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.speech_tokenizer.batch_decode(predicted_ids)[0]
            
            print(f"Transcription: {transcription}")
            return transcription
            
        except Exception as e:
            print(f"Error in speech-to-text conversion: {e}")
            return ""
    
    def text_to_image(self, prompt, output_path="generated_image.png"):
        """
        Generate an image from text using Stable Diffusion
        
        Parameters:
        prompt (str): Text prompt for image generation
        output_path (str): Path to save the generated image
        
        Returns:
        image: Generated image
        """
        try:
            print(f"Generating image from prompt: {prompt}")
            
            # Generate image from prompt
            with torch.autocast(self.device):
                image = self.image_generator(prompt).images[0]
            
            # Save the image
            image.save(output_path)
            print(f"Image saved to {output_path}")
            return image
            
        except Exception as e:
            print(f"Error in text-to-image conversion: {e}")
            return None
    
    def audio_to_image(self, audio_file_path, output_path="generated_image.png"):
        """
        Convert audio to image (speech-to-text followed by text-to-image)
        
        Parameters:
        audio_file_path (str): Path to the audio file
        output_path (str): Path to save the generated image
        
        Returns:
        tuple: (transcription, image)
        """
        # First convert speech to text
        transcription = self.speech_to_text(audio_file_path)
        
        if not transcription:
            print("No transcription generated.")
            return None, None
        
        # Then convert text to image
        image = self.text_to_image(transcription, output_path)
        
        return transcription, image

# For testing
if __name__ == "__main__":
    model = ImageModel()
    print("Models initialized successfully!") 