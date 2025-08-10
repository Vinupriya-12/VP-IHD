import os
import torch
import time
from PIL import Image
import numpy as np
from glob import glob
import streamlit as st
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from diffusers.configuration_utils import FrozenDict

# Initialize CUDA properly
def get_device():
    if torch.cuda.is_available():
        torch.cuda.init()  # Ensure CUDA is properly initialized
        device = torch.device("cuda")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"Using device: {device}")
        return device
    else:
        device = torch.device("cpu")
        print(f"CUDA not available, using CPU")
        print(f"Using device: {device}")
        return device

device = get_device()


# Suppress TensorFlow oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add FrozenDict to safe globals
#torch.serialization.add_safe_globals([FrozenDict])

# Set page config as the FIRST Streamlit command
st.set_page_config(page_title="Interior Design AI Studio", layout="wide")

# Directory setup
OUTPUT_DIR_GENERATE = "generated_interiors"
OUTPUT_DIR_EDIT = "edited_interiors"
os.makedirs(OUTPUT_DIR_GENERATE, exist_ok=True)
os.makedirs(OUTPUT_DIR_EDIT, exist_ok=True)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    /* General styling */
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f5f7fa;
    }
    .stApp {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    h1 {
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 10px;
    }
    h2 {
        font-size: 1.8em;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1em;
        font-weight: 500;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }

    /* Text inputs and text areas */
    .stTextInput>div>input, .stTextArea>textarea {
        border: 1px solid #dcdcdc;
        border-radius: 5px;
        padding: 10px;
    }

    /* Success and info messages */
    .stSuccess, .stInfo {
        border-left: 4px solid #3498db;
        padding: 10px;
        border-radius: 5px;
        background-color: #e6f3ff;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1em;
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
        background-color: #ecf0f1;
        margin-right: 5px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }

    /* Sidebar */
    .stSidebar {
        background-color: #2c3e50;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .stSidebar h3 {
        color: white;
    }
    .stSidebar .stButton>button {
        background-color: #3498db;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Model loading functions
def load_saved_model_for_generation(model_path, state_dict=None, device=None):
    """Load the saved fine-tuned model for text-to-image generation"""
    if device is None:
        device = get_device()  # Use our proper device detection

    # Load base model
    model_id = "bhoomikagp/sd2-interior-model-version2"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )

    # Use provided state_dict or load from file
    if state_dict is None:
        st.info(f"Loading state dictionary from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.warning("Trying to load model with default settings...")
            state_dict = torch.load(model_path, map_location=device)
    
    # Load weights
    if "unet" in state_dict:
        pipe.unet.load_state_dict(state_dict["unet"], strict=False)
    if "vae" in state_dict:
        pipe.vae.load_state_dict(state_dict["vae"], strict=False)
    if "text_encoder" in state_dict:
        pipe.text_encoder.load_state_dict(state_dict["text_encoder"], strict=False)
    if "scheduler_config" in state_dict:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(state_dict["scheduler_config"])
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Move pipeline to device
    pipe = pipe.to(device)
    st.success(f"Generation model loaded successfully to {device}")

    return pipe

def load_saved_model_for_img2img(model_path, state_dict=None, device=None):
    """Load the saved fine-tuned model for image-to-image generation"""
    if device is None:
        device = get_device()  # Use our proper device detection

    # Load base model
    model_id = "bhoomikagp/sd2-interior-model-version2"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )

    # Use provided state_dict or load from file
    if state_dict is None:
        st.info(f"Loading state dictionary from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.warning("Trying to load model with default settings...")
            state_dict = torch.load(model_path, map_location=device)
    
    # Load weights
    if "unet" in state_dict:
        pipe.unet.load_state_dict(state_dict["unet"], strict=False)
    if "vae" in state_dict:
        pipe.vae.load_state_dict(state_dict["vae"], strict=False)
    if "text_encoder" in state_dict:
        pipe.text_encoder.load_state_dict(state_dict["text_encoder"], strict=False)
    if "scheduler_config" in state_dict:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(state_dict["scheduler_config"])
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Move pipeline to device
    pipe = pipe.to(device)
    st.success(f"Img2img model loaded successfully to {device}")

    return pipe

def load_models(model_path):
    """Load both models directly from the .pth file"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Checking model path... 🔍")
        progress_bar.progress(10)
        
        # Verify file exists
        if not os.path.isfile(model_path):
            st.error(f"Error: '{model_path}' is not a valid file. Please provide the correct path.")
            return None, None, f"Error: '{model_path}' is not a valid file. Please provide the correct path."
        
        # Load state_dict once
        status_text.text("Loading model state dictionary... 📂")
        progress_bar.progress(15)
        current_device = get_device()  # Get current device
        try:
            state_dict = torch.load(model_path, map_location=current_device)
            st.info("Model weights loaded successfully")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.warning("Trying to load model with default settings...")
            state_dict = torch.load(model_path, map_location=current_device)
        
        # Load generation pipeline
        status_text.text("Loading generation model... 🎨")
        progress_bar.progress(25)
        generation_pipe = load_saved_model_for_generation(model_path, state_dict=state_dict)
        
        if generation_pipe is None:
            st.error("Failed to load generation pipeline")
            return None, None, "Error: Failed to load generation pipeline"
        
        # Load editing pipeline
        status_text.text("Loading editing model... ✂️")
        progress_bar.progress(75)
        editing_pipe = load_saved_model_for_img2img(model_path, state_dict=state_dict)
        
        if editing_pipe is None:
            st.error("Failed to load editing pipeline")
            return None, None, "Error: Failed to load editing pipeline"
        
        progress_bar.progress(100)
        status_text.text("Models loaded successfully! 🚀")
        
        return generation_pipe, editing_pipe, "Both models loaded successfully! You can now generate or edit interior images."
    
    except Exception as e:
        error_msg = f"Error loading models: {str(e)}"
        st.error(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, error_msg

def preprocess_image(image):
    """Process the image for img2img"""
    if image is None:
        return None

    # Convert from numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Resize if needed
    width, height = image.size
    max_dim = 768  # Max dimension for stable diffusion

    if width > max_dim or height > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))

        image = image.resize((new_width, new_height), Image.LANCZOS)
        st.info(f"Image resized to {new_width}x{new_height}")

    return image

def generate_interior(prompt, guidance_scale, num_inference_steps, generation_pipe):
    """Generate a new interior based on the prompt"""
    
    # Make sure model is loaded
    if generation_pipe is None:
        return None, "Model not loaded. Please load the model first."
    
    st.info(f"Generating interior with prompt: '{prompt}'")
    st.info(f"Parameters: Guidance Scale={guidance_scale}, Steps={num_inference_steps}")
    
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress
        status_text.text("Starting generation... 🎨")
        progress_bar.progress(10)
        
        # Generate image with GPU acceleration if available
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda"):
                    result = generation_pipe(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps
                    ).images[0]
            else:
                result = generation_pipe(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]
        
        progress_bar.progress(100)
        status_text.text("Generation complete! ✅")
        
        # Save result
        timestamp = int(time.time())
        result_path = os.path.join(OUTPUT_DIR_GENERATE, f"generated_interior_{timestamp}.png")
        result.save(result_path)
        
        return result, f"Image generated and saved to {result_path}"
    
    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        st.error(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg

def edit_interior(image, edit_prompt, strength, guidance_scale, num_inference_steps, editing_pipe):
    """Edit an existing interior image based on the prompt"""
    
    # Make sure model is loaded
    if editing_pipe is None:
        return None, "Model not loaded. Please load the model first."
    
    if image is None:
        return None, "No image provided. Please upload an image first."
    
    try:
        # Preprocess image
        init_image = preprocess_image(image)
        
        # Add a prefix to ensure we maintain the interior context
        full_prompt = f"An interior design with {edit_prompt}"
        
        st.info(f"Editing interior with prompt: '{full_prompt}'")
        st.info(f"Parameters: Strength={strength}, Guidance Scale={guidance_scale}, Steps={num_inference_steps}")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress
        status_text.text("Starting image editing... ✂️")
        progress_bar.progress(10)
        
        # Generate the edited image with GPU acceleration if available
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda"):
                    result = editing_pipe(
                        prompt=full_prompt,
                        image=init_image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps
                    ).images[0]
            else:
                result = editing_pipe(
                    prompt=full_prompt,
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]
        
        progress_bar.progress(100)
        status_text.text("Editing complete! ✅")
        
        # Save result
        timestamp = int(time.time())
        result_path = os.path.join(OUTPUT_DIR_EDIT, f"edited_interior_{timestamp}.png")
        result.save(result_path)
        
        return result, f"Image edited and saved to {result_path}"
    
    except Exception as e:
        error_msg = f"Error during editing: {str(e)}"
        st.error(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg

def get_example_prompts():
    """Return example prompts for generation"""
    return [
        "A modern minimalist living room with large windows and natural light",
        "A cozy rustic bedroom with wooden beams and a fireplace",
        "A luxurious bathroom with marble flooring and a freestanding bathtub",
        "A Scandinavian style kitchen with white cabinets and wooden countertops",
        "An industrial loft apartment with exposed brick walls and high ceilings"
    ]

def get_example_edits():
    """Return example editing instructions"""
    return [
        "Change the wall color to light blue",
        "Add wooden flooring",
        "Make it more minimalist",
        "Transform to Scandinavian style",
        "Add more plants and greenery"
    ]

def main():
    # Banner
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src='https://via.placeholder.com/1200x200/3498db/ffffff?text=Interior+Design+AI+Studio' style='width: 100%; border-radius: 10px;'>
        </div>
    """, unsafe_allow_html=True)
    
    st.title("🏠 Interior Design AI Studio")
    st.markdown("Transform your space with AI-powered interior design generation and editing.")

    # Sidebar for setup
    with st.sidebar:
        st.header("🛠️ Model Setup")
        model_path = st.text_input(
            "Model Path", 
            value=r"C:\Users\alsto\Desktop\gen ai\gen ai project-20250421T045900Z-002\gen ai project\final_interior_model_20250416_103011.pth",
            help="Path to your trained model file (.pth)",
            key="model_path_input"
        )
        
        force_weights_only_false = st.checkbox(
            "Force loading with weights_only=False", 
            value=True,
            help="Enable this if you're having issues loading the model",
            key="force_weights_only_checkbox"
        )
        
        if st.button("Load Models 🚀", key="load_models_button"):
            
            st.info("Loading models with weights_only=False...")
            generation_pipe, editing_pipe, model_status = load_models(model_path)
            
            # Debug: Display model_status
            st.write(f"Debug: model_status = {model_status}")
            
            # Store in session state
            st.session_state.generation_pipe = generation_pipe
            st.session_state.editing_pipe = editing_pipe
            st.session_state.model_status = model_status
            
            if generation_pipe is not None and editing_pipe is not None:
                st.session_state.model_loaded = True
            else:
                st.session_state.model_loaded = False
        
        # Model status in sidebar
        st.subheader("📊 Model Status")
        st.text_area("", value=st.session_state.get("model_status", ""), height=100, disabled=True, key="model_status_textarea")
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("Create stunning interior designs with AI. Generate new designs or edit existing ones with ease.")

    # Initialize session state
    if "generation_pipe" not in st.session_state:
        st.session_state.generation_pipe = None
    if "editing_pipe" not in st.session_state:
        st.session_state.editing_pipe = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "model_status" not in st.session_state:
        st.session_state.model_status = ""

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Generate Interior", "✂️ Edit Interior", "🖼️ Gallery", "💡 Examples"])

    with tab1:
        st.header("🎨 Generate New Interior Design")
        st.markdown("Create stunning interiors from your imagination.")
        
        if not st.session_state.model_loaded:
            st.warning("Please load the model in the sidebar first.")
        else:
            with st.container():
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    gen_prompt = st.text_area(
                        "Prompt", 
                        placeholder="A modern minimalist living room with large windows and natural light",
                        height=100,
                        key="gen_prompt_textarea"
                    )
                    
                    gen_guidance = st.slider(
                        "Guidance Scale", 
                        min_value=1.0, max_value=15.0, value=7.5, 
                        help="Higher values make the image adhere more to the prompt",
                        key="gen_guidance_slider"
                    )
                    
                    gen_steps = st.slider(
                        "Inference Steps", 
                        min_value=20, max_value=100, value=50, step=1,
                        help="More steps improve quality but take longer",
                        key="gen_steps_slider"
                    )
                    
                    if st.button("Generate Interior 🎨", key="generate_interior_button"):
                        with st.spinner("Crafting your interior design..."):
                            result_img, result_msg = generate_interior(
                                gen_prompt, 
                                gen_guidance, 
                                gen_steps, 
                                st.session_state.generation_pipe
                            )
                            
                            if result_img is not None:
                                st.session_state.gen_result = result_img
                                st.session_state.gen_message = result_msg
                            else:
                                st.session_state.gen_message = result_msg
                
                with col2:
                    if "gen_result" in st.session_state and st.session_state.gen_result is not None:
                        st.image(st.session_state.gen_result, caption="Generated Interior", use_container_width=True)  # Changed from use_column_width
                        st.markdown(f"**Status**: {st.session_state.gen_message}")
                    else:
                        st.image("https://via.placeholder.com/400x300/ecf0f1/7f8c8d?text=Generated+Image+Preview", caption="Generated image will appear here", use_container_width=True)  # Changed from use_column_width

    with tab2:
        st.header("✂️ Edit Existing Interior Design")
        st.markdown("Upload an image and transform it with your vision.")
        
        if not st.session_state.model_loaded:
            st.warning("Please load the model in the sidebar first.")
        else:
            with st.container():
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    edit_image_input = st.file_uploader("Upload Interior Image", type=["png", "jpg", "jpeg"], key="edit_image_uploader")
                    
                    if edit_image_input is not None:
                        input_image = Image.open(edit_image_input)
                        st.image(input_image, caption="Uploaded Image", use_container_width=True)  # Changed from use_column_width
                    
                    edit_prompt = st.text_area(
                        "Editing Instructions", 
                        placeholder="Change the wall color to light blue",
                        height=100,
                        key="edit_prompt_textarea"
                    )
                    
                    edit_strength = st.slider(
                        "Edit Strength", 
                        min_value=0.1, max_value=1.0, value=0.7, step=0.05,
                        help="Higher values apply more changes",
                        key="edit_strength_slider"
                    )
                    
                    edit_guidance = st.slider(
                        "Guidance Scale", 
                        min_value=1.0, max_value=15.0, value=7.5, 
                        help="Higher values make the image adhere more to the prompt",
                        key="edit_guidance_slider"
                    )
                    
                    edit_steps = st.slider(
                        "Inference Steps", 
                        min_value=20, max_value=100, value=50, step=1,
                        help="More steps improve quality but take longer",
                        key="edit_steps_slider"
                    )
                    
                    if st.button("Edit Interior ✂️", key="edit_interior_button") and edit_image_input is not None:
                        with st.spinner("Transforming your interior design..."):
                            result_img, result_msg = edit_interior(
                                input_image, 
                                edit_prompt, 
                                edit_strength, 
                                edit_guidance, 
                                edit_steps, 
                                st.session_state.editing_pipe
                            )
                            
                            if result_img is not None:
                                st.session_state.edit_result = result_img
                                st.session_state.edit_message = result_msg
                            else:
                                st.session_state.edit_message = result_msg
                
                with col2:
                    if "edit_result" in st.session_state and st.session_state.edit_result is not None:
                        st.image(st.session_state.edit_result, caption="Edited Interior", use_container_width=True)  # Changed from use_column_width
                        st.markdown(f"**Status**: {st.session_state.edit_message}")
                    else:
                        st.image("https://via.placeholder.com/400x300/ecf0f1/7f8c8d?text=Edited+Image+Preview", caption="Edited image will appear here", use_container_width=True)  # Changed from use_column_width

    with tab3:
        st.header("🖼️ Gallery")
        st.markdown("Explore your generated and edited interiors.")
        
        with st.container():
            if st.button("Refresh Gallery 🔄", key="refresh_gallery_button"):
                st.session_state.refresh_gallery = True
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Generated Interiors")
                generated_images = sorted(glob(os.path.join(OUTPUT_DIR_GENERATE, "*.png")), key=os.path.getmtime, reverse=True)
                
                if generated_images:
                    for img_path in generated_images:
                        img = Image.open(img_path)
                        timestamp = os.path.basename(img_path).split('_')[-1].split('.')[0]
                        st.image(img, caption=f"Generated on: {time.ctime(int(timestamp))}", use_container_width=True)  # Changed from use_column_width
                else:
                    st.info("No generated images found. Try generating one!")
            
            with col2:
                st.subheader("Edited Interiors")
                edited_images = sorted(glob(os.path.join(OUTPUT_DIR_EDIT, "*.png")), key=os.path.getmtime, reverse=True)
                
                if edited_images:
                    for img_path in edited_images:
                        img = Image.open(img_path)
                        timestamp = os.path.basename(img_path).split('_')[-1].split('.')[0]
                        st.image(img, caption=f"Edited on: {time.ctime(int(timestamp))}", use_container_width=True)  # Changed from use_column_width
                else:
                    st.info("No edited images found. Try editing one!")

    with tab4:
        st.header("💡 Examples")
        st.markdown("Get inspired with example prompts and editing instructions.")
        
        with st.container():
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Example Prompts for Generation")
                example_prompts = get_example_prompts()
                for i, prompt in enumerate(example_prompts):
                    if st.button(f"Example {i+1} 📝", key=f"gen_example_{i}"):
                        st.session_state.selected_gen_prompt = prompt
                        st.rerun()
                
                st.markdown("**Examples**:")
                for prompt in example_prompts:
                    st.markdown(f"- {prompt}")
            
            with col2:
                st.subheader("Example Editing Instructions")
                example_edits = get_example_edits()
                for i, prompt in enumerate(example_edits):
                    if st.button(f"Example {i+1} ✏️", key=f"edit_example_{i}"):
                        st.session_state.selected_edit_prompt = prompt
                        st.rerun()
                
                st.markdown("**Examples**:")
                for edit in example_edits:
                    st.markdown(f"- {edit}")

    # Notes in an expander
    with st.expander("ℹ️ About & Notes"):
        st.markdown("""
        ### About
        Interior Design AI Studio lets you create and edit stunning interior designs using AI. Powered by Stable Diffusion, it supports text-to-image generation and image-to-image editing.

        ### Notes
        - **Model Loading**: Load the model in the sidebar before generating or editing.
        - **Model File**: The app uses a `.pth` file with all necessary components.
        - **Troubleshooting**: If model loading fails, enable "Force loading with weights_only=False".
        - **Generation**: Create new interiors from text descriptions.
        - **Editing**: Modify existing images with text instructions.
        - **Quality**: Higher inference steps improve quality but take longer.
        - **Storage**: Images are saved in 'generated_interiors' and 'edited_interiors' folders.
        - **Gallery**: View all your creations in the Gallery tab.
        """)

# Check if running in Google Colab and mount drive if needed
try:
    from google.colab import drive
    print("Running in Google Colab, mounting drive...")
    drive.mount("/content/drive", force_remount=False)
except (ImportError, AttributeError):
    print("Not running in Google Colab, skipping drive mount")

# Run the app
if __name__ == "__main__":
    main()
