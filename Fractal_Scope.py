import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Function to generate Mandelbrot Set
def mandelbrot(c, max_iter):
    z = c
    for i in range(max_iter):
        if abs(z) > 2:
            return i
        z = z * z + c
    return max_iter

# Function to generate Julia Set
def julia(c, max_iter, x_min, x_max, y_min, y_max, width, height):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    image = np.zeros((height, width))

    for ix in range(width):
        for iy in range(height):
            z = x[ix] + 1j * y[iy]
            image[iy, ix] = mandelbrot(z, max_iter)
    
    return image

# Function to generate Mandelbrot Set image
def generate_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    image = np.zeros((height, width))

    for ix in range(width):
        for iy in range(height):
            c = x[ix] + 1j * y[iy]
            image[iy, ix] = mandelbrot(c, max_iter)

    return image

# Streamlit App
st.title("Interactive Mandelbrot and Julia Set Visualizer")
st.sidebar.header("Configuration Options")

# Sidebar options
set_type = st.sidebar.radio("Select Set to Visualize", ["Mandelbrot Set", "Julia Set"])

# Common parameters
x_min = st.sidebar.number_input("x_min", value=-2.0, step=0.1)
x_max = st.sidebar.number_input("x_max", value=1.0, step=0.1)
y_min = st.sidebar.number_input("y_min", value=-1.5, step=0.1)
y_max = st.sidebar.number_input("y_max", value=1.5, step=0.1)
width = st.sidebar.slider("Image Width (px)", min_value=100, max_value=2000, value=800, step=100)
height = st.sidebar.slider("Image Height (px)", min_value=100, max_value=2000, value=800, step=100)
max_iter = st.sidebar.slider("Max Iterations", min_value=50, max_value=1000, value=256, step=50)

# Julia Set specific parameter
if set_type == "Julia Set":
    real_part = st.sidebar.number_input("Julia Constant - Real Part", value=-0.7, step=0.1)
    imag_part = st.sidebar.number_input("Julia Constant - Imaginary Part", value=0.27015, step=0.1)
    julia_constant = complex(real_part, imag_part)

# Plot the selected set
if st.button("Generate Plot"):
    if set_type == "Mandelbrot Set":
        st.write("### Mandelbrot Set")
        mandelbrot_image = generate_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mandelbrot_image, cmap="hot", extent=(x_min, x_max, y_min, y_max))
        ax.set_title("Mandelbrot Set")
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
        st.pyplot(fig)

        # Provide download link
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("Download Mandelbrot Set Image", data=buf, file_name="mandelbrot_set.png", mime="image/png")

    elif set_type == "Julia Set":
        st.write("### Julia Set")
        julia_image = julia(julia_constant, max_iter, x_min, x_max, y_min, y_max, width, height)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(julia_image, cmap="hot", extent=(x_min, x_max, y_min, y_max))
        ax.set_title(f"Julia Set for c = {julia_constant}")
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        st.pyplot(fig)

        # Provide download link
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("Download Julia Set Image", data=buf, file_name="julia_set.png", mime="image/png")

st.sidebar.markdown("### Instructions")
st.sidebar.markdown(
    """
    - Use the sliders and inputs to customize the parameters.
    - Click **Generate Plot** to view the fractal.
    - For Julia Set, specify the constant `c`.
    - Download the generated plot as a PNG image.
    """
)
