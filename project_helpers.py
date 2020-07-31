import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

def pad_image(grayscale_image, window_size):
    """Pad the borders of an image with zeros.
    
    Args:
        grayscale_image (np.ndarray): Input image.
        window_size (int): Size of the window we are blurring or convolving over.
            Must be odd.
    
    Returns:
        np.ndarray: Zero-padded image.
    """
    assert len(grayscale_image.shape) == 2, "Image shape must be (rows, cols)"
    assert window_size % 2 == 1, "Window size must be odd!"
    pad_amount = window_size // 2

    # Note: we've explicitly written out the pad operation here to be explicit,
    # but we could also use the built-in `np.pad()` function
    padded_image = np.zeros(
        shape=(
            grayscale_image.shape[0] + pad_amount * 2,
            grayscale_image.shape[1] + pad_amount * 2,
        )
    )
    padded_image[
        pad_amount:-pad_amount, pad_amount:-pad_amount
    ] = grayscale_image

    return padded_image


def normalize(input_array):
    """Normalize an array. Takes an array with arbitrary values, and rescales
    them to sit between 0.0 and 1.0.
    
    Args:
        input_array (np.ndarray): Array to normalize.

    Returns:
        np.ndarray: Normalized array.
    """
    output_image = input_array - np.min(input_array)
    output_image = output_image / np.max(output_image)
    return output_image

def histogram_single_channel(image, bins):
    """Given a single-channel image (2D array) and bin count, compute a
    corresponding image histogram.
    
    Args:
        image (np.ndarray): Input image. Shape should be (rows, cols).
        bins (int): Number of bins in our image histogram.
    Returns:
        np.ndarray: 1D NumPy array containing our histogram values.
    """
    # Validate input
    assert isinstance(image, np.ndarray), "Input should be a NumPy array!"
    assert len(image.shape) == 2, "Shape of image should be (rows, cols)"
    #assert np.min(image) >= 0.0, "Intensities should be between 0.0 and 1.0!"
    #assert np.max(image) <= 1.0, "Intensities should be between 0.0 and 1.0!"

    # Get image histogram and bin edges
    hist, bin_edges = np.histogram(
        a=image,
        bins=bins,
        range=(0.0, 1.0),
    )

    # Return histogram
    return hist

def visualize_labels(title, images, labels, seed=42):
    """Visualize a few images from our dataset!
    
    Args:
        title (str): Title of figure.
        images (np.ndarray): An array containing images. Shape should be (N, 64, 64, 3).
        labels (np.ndarray): An array of labels, 0 or 1. Shape should be (N, )
        seed (int): Seed used to generate random samples. Change this to see different samples!
    """
    # Validate inputs
    assert len(images.shape) == 4
    assert len(labels.shape) == 1
    assert labels.shape[0] == images.shape[0]
    assert images.shape[-1] == 3

    generator = np.random.RandomState(seed)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for ax in axes.flatten():
        index = generator.randint(len(images))
        ax.imshow(images[index])
        if labels[index] == 1:
            ax.set_title(f"$\\bf{{Safe}}$ ({index})")
        else:
            ax.set_title(f"({index})")
    fig.suptitle(title, fontsize=16)

def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    
    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!
    # ***** Start of your code *****
    row_scale_factor = input_rows/output_rows
    col_scale_factor = input_cols/output_cols
    
    for i in range(len(output_image)):
        for j in range(len(output_image[i])):
            input_i = int(i * row_scale_factor)
            input_j = int(j * col_scale_factor)
            output_image[i, j, :] = input_image[input_i, input_j, :]
    
    # ***** End of your code *****

    # 3. Return the output image
    return output_image

from cv_helpers import rgb2hsl, hsl2rgb

def tint_red(rgb_img):
    red_array = np.array([[0]*300]*300)
    hsl_img = rgb2hsl(rgb_img)
    
    hsl_img[:, :, 0] = hsl_img[:, :, 0] * red_array
    output_image = hsl2rgb(hsl_img)
    return output_image

def tint_green(rgb_img):
    green_array = np.array([[2.5]*300]*300)
    hsl_img = rgb2hsl(rgb_img)
    h_channel = hsl_img[:, :, 0]
    hsl_img[:, :, 0] = h_channel * green_array
    output_image = hsl2rgb(hsl_img)
    return output_image