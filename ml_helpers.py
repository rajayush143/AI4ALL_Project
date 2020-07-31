import time

import numpy as np
import PIL.Image
import yaml


def load_dataset():
    """Load our aerial imagery dataset.

    Note that this function currently downscales all images from 300x300 to
    128x128 -- this makes them a bit easier to work with, but you're welcome to
    try and use the full images to try and improve performance.

    If you do do* this, note that you'll need 5.5x more memory to store the same
    number of images! To make things more manageable, you can try removing the
    `/ 255.0` normalization term and storing the images as type `np.uint8` instead
    of `np.float32`. (this reduces memory usage by 4x)

    *lol

    Returns:
        images (np.ndarray): Array of images. Shape should be (N, 128, 128, 3).
        labels (np.ndarray): Array of labels, 0 or 1. Shape should be (N,).
    """
    # Get the start time
    start_time = time.time()

    # Load dataset YAML file
    # This contains all of our image labels, as well as locations of the images themself
    print("Reading dataset/dataset.yaml... ", end="")
    with open("dataset/dataset.yaml", "r") as file:
        dataset = yaml.safe_load(file)

    # Get paths, labels
    paths = []
    labels = []
    for sample in dataset:
        # Assign a "1" label if we're looking at the ground
        # 0 for everything else: trees, buildings, cars, etc
        label_semantic = max(sample["labels"].keys(), key=sample["labels"].get)
        if max(sample["labels"].values()) < 0.80:
            # Samples that are not obviously in any one category: unsafe
            label=0
        elif label_semantic == "GROUND":
            # Safe if >80% ground
            label = 1
        else:
            # Unsafe otherwise, this is usually water
            label = 0

        paths.append(sample["path"])
        labels.append(label)
    print("done!", flush=True)

    print("Loading images", end="")
    # Get images
    images = np.zeros((len(paths), 128, 128, 3), dtype=np.float32)
    progress = 0.0
    for i, path in enumerate(paths):
        images[i] = np.array(PIL.Image.open(path).resize((128, 128))) / 255.0
        if i / len(paths) > progress:
            progress += 1.0 / 20.0
            print(".", end="", flush=True)
    print(" done!")
    labels = np.array(labels, dtype=np.int)

    # Return
    print(f"Loaded {len(images)} images in {time.time() - start_time} seconds!")
    return images, labels