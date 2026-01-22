import streamlit as st
import torch
import numpy as np
import cv2
import imageio
from PIL import Image, ImageOps
from vae_model import ConvVAE
from torchvision import datasets, transforms
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Neural Morph Ultimate", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_model():
    model = ConvVAE().to(DEVICE)
    if os.path.exists('models/vae_mnist_conv.pth'):
        model.load_state_dict(torch.load('models/vae_mnist_conv.pth', map_location=DEVICE))
    else:
        st.error("Model not found! Please run train.py first.")
    model.eval()
    return model

@st.cache_resource
def load_data():
    return datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

def get_digit_from_db(target_label, dataset):
    """Finds a digit in the MNIST database"""
    for img, label in dataset:
        if label == target_label:
            return img.unsqueeze(0).to(DEVICE)
    return dataset[0][0].unsqueeze(0).to(DEVICE)

def process_uploaded_image(uploaded_file, invert=True):
    try:
        img = Image.open(uploaded_file).convert('L')
        if invert:
            img = ImageOps.invert(img)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        transform = transforms.ToTensor()
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        return tensor
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def upscale_frame(frame, scale=8):
    h, w = frame.shape
    return cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

def slerp(val, low, high):
    """Spherical Linear Interpolation"""
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    dot = torch.sum(low_norm * high_norm, dim=1, keepdim=True)
    dot = torch.clamp(dot, -1, 1)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.item() < 1e-4:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high

# --- SIDEBAR UI ---
st.sidebar.title("Controls")
mode = st.sidebar.radio("Input Source", ["Database (Multi-Digit)", "Upload Images (Single)"])

st.sidebar.markdown("---")
interpolation_mode = st.sidebar.radio("Math Mode", ["SLERP (Spherical)", "Linear (Standard)"])
steps = st.sidebar.slider("Frames", 10, 100, 45)
fps = st.sidebar.slider("Speed (FPS)", 10, 60, 30)
scale = st.sidebar.selectbox("Resolution Scale", [4, 8, 16], index=1)

# --- MAIN UI ---
st.title("Neural Morph Version 3")

try:
    model = load_model()
    if mode == "Database (Multi-Digit)":
        dataset = load_data()
except Exception as e:
    st.stop()

# --- INPUT HANDLING ---
mu_sources = [] # List of latent vectors
mu_targets = []
ready_to_morph = False

if mode == "Database (Multi-Digit)":
    st.info("Enter multi-digit numbers (e.g., '1990' to '2025'). Both must have the same length.")
    c1, c2 = st.columns(2)
    with c1:
        source_str = st.text_input("Start Number", "12")
    with c2:
        target_str = st.text_input("End Number", "34")
    
    if st.button("Load Digits & Morph"):
        if len(source_str) != len(target_str):
            st.error("Error: Numbers must have the same number of digits!")
        else:
            with torch.no_grad():
                # Loop through each character string, find it in DB, and encode it
                for s_char, t_char in zip(source_str, target_str):
                    if not (s_char.isdigit() and t_char.isdigit()):
                        st.error("Please enter only digits (0-9).")
                        st.stop()
                        
                    s_img = get_digit_from_db(int(s_char), dataset)
                    t_img = get_digit_from_db(int(t_char), dataset)
                    
                    mu_sources.append(model.encode(s_img)[0])
                    mu_targets.append(model.encode(t_img)[0])
            ready_to_morph = True

elif mode == "Upload Images (Single)":
    st.warning("Upload mode currently only supports single digits (the model can't split images yet).")
    invert_chk = st.checkbox("Invert Colors", value=True)
    
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("Start Image", type=['png', 'jpg', 'jpeg'])
        if f1: st.image(f1, width=100)
    with c2:
        f2 = st.file_uploader("End Image", type=['png', 'jpg', 'jpeg'])
        if f2: st.image(f2, width=100)
        
    if f1 and f2 and st.button("Process & Morph"):
        t1 = process_uploaded_image(f1, invert_chk)
        t2 = process_uploaded_image(f2, invert_chk)
        
        if t1 is not None and t2 is not None:
            with torch.no_grad():
                mu_sources.append(model.encode(t1)[0])
                mu_targets.append(model.encode(t2)[0])
            ready_to_morph = True

# --- GENERATION LOGIC ---
if ready_to_morph and len(mu_sources) > 0:
    progress_bar = st.progress(0)
    frames = []
    
    t_values = np.linspace(0, 1, steps)
    t_values = np.concatenate([t_values, t_values[::-1]]) 

    for i, t in enumerate(t_values):
        progress_bar.progress(int((i / len(t_values)) * 100))
        t_smooth = t * t * (3 - 2 * t) 
        
        # Calculate individual digit frames
        digit_frames = []
        for mu_s, mu_t in zip(mu_sources, mu_targets):
            if "SLERP" in interpolation_mode:
                z = slerp(t_smooth, mu_s, mu_t)
            else:
                z = mu_s + t_smooth * (mu_t - mu_s)
            
            with torch.no_grad():
                recon = model.decode(z).cpu().squeeze().numpy()
                recon_hd = upscale_frame(recon, scale=scale)
                digit_frames.append(recon_hd)
        
        # Stitch them together horizontally (Stitching Logic)
        combined_frame = np.hstack(digit_frames)
        
        frame_uint8 = (np.clip(combined_frame, 0, 1) * 255).astype(np.uint8)
        frames.append(frame_uint8)

    # --- DISPLAY ---
    gif_path = "morph_output.gif"
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    
    st.divider()
    res_col1, res_col2 = st.columns([3, 2])
    with res_col1:
        st.subheader("Result")
        st.image(gif_path, use_container_width=True)
    with res_col2:
        st.subheader("Export")
        st.success("Morph Generated!")
        with open(gif_path, "rb") as f:
            st.download_button("Download GIF", f, "neural_morph.gif", "image/gif")