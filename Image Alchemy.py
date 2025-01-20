import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import io

# --- Basic Filters ---
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# --- Transformations ---
def scale_image(image, scale_factor=1.5):
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

def rotate_image(image, angle=45):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def flip_image(image, flip_code=1):  # 0 for vertical, 1 for horizontal
    return cv2.flip(image, flip_code)

# --- Custom Filters ---
def apply_color_filter(image, r_factor=1.0, g_factor=1.0, b_factor=1.0):
    b, g, r = cv2.split(image)
    r = np.clip(r * r_factor, 0, 255).astype(np.uint8)
    g = np.clip(g * g_factor, 0, 255).astype(np.uint8)
    b = np.clip(b * b_factor, 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# --- Fetch Random Image from Web ---
def fetch_random_image():
    url = "https://picsum.photos/500/500"
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    return np.array(img)

# --- Streamlit UI ---
st.title("Image Filter Application")

if 'random_image' not in st.session_state:
    st.session_state.random_image = None

if 'filter_history' not in st.session_state:
    st.session_state.filter_history = []

image_option = st.radio("Choose Image Source", ["Random Image", "Upload Image"])

if image_option == "Random Image":
    if st.session_state.random_image is None:
        st.write("Fetching a random image from the web...")
        st.session_state.random_image = fetch_random_image()
    image = st.session_state.random_image
    st.image(image, caption="Random Image", use_container_width=True)
    
elif image_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

# --- Filters and Transformations ---
st.sidebar.title("Apply Filters")

filter_choice = st.sidebar.selectbox("Choose Filter", ["None", "Grayscale", "Blur", "Sharpen", "Color Filter", "Edge Detection", "Scale", "Rotate", "Flip"])

if filter_choice != "None":
    if filter_choice == "Grayscale":
        image = apply_grayscale(image)
        st.session_state.filter_history.append("Grayscale")
        
    elif filter_choice == "Blur":
        kernel_size = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2)
        image = apply_blur(image, (kernel_size, kernel_size))
        st.session_state.filter_history.append(f"Blur (Kernel Size: {kernel_size})")
        
    elif filter_choice == "Sharpen":
        image = apply_sharpen(image)
        st.session_state.filter_history.append("Sharpen")
        
    elif filter_choice == "Color Filter":
        r_factor = st.sidebar.slider("Red Factor", 0.0, 2.0, 1.0)
        g_factor = st.sidebar.slider("Green Factor", 0.0, 2.0, 1.0)
        b_factor = st.sidebar.slider("Blue Factor", 0.0, 2.0, 1.0)
        image = apply_color_filter(image, r_factor, g_factor, b_factor)
        st.session_state.filter_history.append(f"Color Filter (R: {r_factor}, G: {g_factor}, B: {b_factor})")
        
    elif filter_choice == "Edge Detection":
        image = edge_detection(image)
        st.session_state.filter_history.append("Edge Detection")
        
    elif filter_choice == "Scale":
        scale_factor = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.5)
        image = scale_image(image, scale_factor)
        st.session_state.filter_history.append(f"Scale (Factor: {scale_factor})")
        
    elif filter_choice == "Rotate":
        angle = st.sidebar.slider("Rotation Angle", 0, 360, 45)
        image = rotate_image(image, angle)
        st.session_state.filter_history.append(f"Rotate (Angle: {angle})")
        
    elif filter_choice == "Flip":
        flip_code = st.sidebar.radio("Flip Direction", [0, 1])
        image = flip_image(image, flip_code)
        direction = 'Vertical' if flip_code == 0 else 'Horizontal'
        st.session_state.filter_history.append(f"Flip ({direction})")

st.image(image.astype(np.uint8), caption="Processed Image", use_container_width=True)

# --- Save and Download Image ---
save_option = st.sidebar.button("Save Image")
if save_option:
    output_image_path = "output_image.jpg"
    output_image = Image.fromarray(image.astype(np.uint8))
    output_image.save(output_image_path)
    
    # Provide a download link
    with open(output_image_path,"rb") as f:
        btn = st.download_button(
            label="Download Processed Image",
            data=f,
            file_name=output_image_path,
            mime="image/jpeg"
        )
    
st.sidebar.title("Filter History")
for idx in range(len(st.session_state.filter_history)):
    st.sidebar.write(f"{idx + 1}. {st.session_state.filter_history[idx]}")

# --- Clear History ---
if st.sidebar.button("Clear History"):
    st.session_state.filter_history.clear()
