"""Some helper functions for our AI4ALL Computer Vision notebooks.

We encourage you to read through these, and try to understand how each function is
implemented!
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


def visualize_single_channel(title, image):
    """Visualize a 2D array using matplotlib.
    All inputs values should be normalized 0.0-1.0.

    Args:
        title (str): Name of image we're visualizing.
        image (np.ndarray): Image we're visualizing. Shape should be `(rows, cols)`.
    """
    assert type(title) == str, "Title not a string!"
    assert len(image.shape) == 2, "Image array not 2D!"

    # Visualize image
    # We manually set the black value with `vmin`, and the white value with `vmax`
    plt.imshow(image, vmin=0.0, vmax=1.0, cmap="gray")

    # Give our plot a title -- this is purely cosmetic!
    plt.title(f"{title}, shape={image.shape}")

    # Show image
    plt.show()


def visualize_rgb(title, image):
    """Visualize an RGB image using matplotlib.

    Args:
        title (str): Name of image we're visualizing.
        image (np.ndarray): Image we're visualizing. Shape should be `(rows, cols, 3)`.
    """
    assert type(title) == str, "Title not a string!"
    assert len(image.shape) == 3, "Image array not 3D!"
    assert image.shape[2], "Last dimension must have length 3! (RGB)"

    # Visualize image
    plt.imshow(image)

    # Give our plot a title -- this is purely cosmetic!
    plt.title(f"{title}, shape={image.shape}")

    # Show image
    plt.show()


def load_image(path):
    """Load an image from a path.

    Args:
        path (str): Location of image to load.

    Returns:
        np.ndarray: RGB image, with channels normalized from 0.0-1.0. Shape
        should be `(rows, cols, 3)`.
    """
    # Load our image with the Python Image Library
    # Values by default are 0-255, we divide to normalize to the 0.0-1.0 range
    return np.array(PIL.Image.open(path).convert("RGB")) / 255.0


def rgb2hsl(rgb_image):
    """Converts an RGB image to HSL. All channel ranges should be between 0.0 and 1.0.

    Args:
        rgb_image (np.ndarray): Input image, with colors in RGB. Shape should be (rows, cols, 3).

    Returns:
        np.ndarray: Output image, with colors in HSL. Shape should be (rows, cols, 3).
    """
    assert len(rgb_image.shape) == 3, "Image array not 3D!"
    assert rgb_image.shape[2], "Last dimension must have length 3! (RGB)"

    # Use OpenCV to convert RGB image to HLS
    output = cv2.cvtColor(rgb_image.astype(np.float32), cv2.COLOR_RGB2HLS)

    # Flip channels: HLS -> HSL
    output[:, :, 1], output[:, :, 2] = (
        output[:, :, 2].copy(),
        output[:, :, 1].copy(),
    )

    # Normalize H range: 0-360 -> 0-1
    output[:, :, 0] /= 360.0

    # Return HSL image
    return output


def hsl2rgb(hsl_image):
    """Converts an HSL image to RGB. All channel ranges should be between 0.0 and 1.0.

    Args:
        hsl_image (np.ndarray): Input image, with colors in HSL. Shape should be
            `(rows, cols, 3)`.

    Returns:
        np.ndarray: Output image, with colors in RGB. Shape should be
        `(rows, cols, 3)`.
    """
    assert len(hsl_image.shape) == 3, "Image array not 3D!"
    assert hsl_image.shape[2], "Last dimension must have length 3! (HSL)"

    # Create HLS representation of our HSL image
    hls_image = np.zeros_like(hsl_image)

    # Unnormalize H range: 0-1 -> 0-360
    hls_image[:, :, 0] = hsl_image[:, :, 0] * 360.0

    # Flip channels: HSL -> HLS
    hls_image[:, :, 1], hls_image[:, :, 2] = (
        hsl_image[:, :, 2],
        hsl_image[:, :, 1],
    )

    # Use OpenCV to convert HSL image to RGB
    output = cv2.cvtColor(hls_image.astype(np.float32), cv2.COLOR_HLS2RGB)

    # Return RGB image
    return output
