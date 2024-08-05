import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from skimage import io

# Load the pre-trained HOG model for pedestrian detection
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import integral_image

def detect_pedestrians(image_path):
    # Load image
    image = io.imread(image_path)
    
    # Convert image to grayscale
    gray_image = rgb2gray(image)

    # Compute HOG features and HOG image
    hog_features, hog_image = hog(gray_image, pixels_per_cell=(16, 16),
                                  cells_per_block=(2, 2), visualize=True,
                                  multichannel=False)

    # Enhance the contrast of the HOG image for visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Display the original image and HOG image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax[1].set_title('HOG Image')
    ax[1].axis('off')

    plt.show()

# Example usage
detect_pedestrians('path_to_your_image.jpg')
