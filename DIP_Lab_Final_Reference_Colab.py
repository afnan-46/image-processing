# ============================================================
# DIGITAL IMAGE PROCESSING LAB FINAL - COLAB CODE REFERENCE
# Labs 2-7 + Practice + Low/High Pass Filtering
# Run each section as a separate Colab cell if you prefer.
# ============================================================

# ============================================================
# SECTION 1 - INSTALLATION AND IMPORTS
# ============================================================
# Run this first in Google Colab.

!pip -q install seaborn-image

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageOps

import skimage
import skimage as ski
from skimage import data, color, exposure, filters, segmentation
from skimage.util import random_noise
from skimage.exposure import match_histograms
from skimage.color import label2rgb
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.morphology import erosion, dilation
from scipy.ndimage import convolve, median_filter, label
from scipy.fftpack import dct, idct

import seaborn_image as isns


# ============================================================
# SECTION 2 - UTILITY FUNCTION: SHOW MULTIPLE IMAGES
# ============================================================
# Use this function to display images in a labelled grid.

def img_grid(images, title='Image Plot', subtitles=None, cols=3, figsize=(18, 10), cmap='gray'):
    if subtitles is None:
        subtitles = [''] * len(images)

    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    fig.suptitle(title, fontsize=16, y=1.02)

    for ax, image, subtitle in zip(axes, images, subtitles):
        arr = np.array(image) if isinstance(image, Image.Image) else image
        if arr.ndim == 2:
            ax.imshow(arr, cmap=cmap)
        else:
            ax.imshow(arr)
        ax.set_title(subtitle)
        ax.axis('off')

    for ax in axes[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
# SECTION 3 - LOAD IMAGE FROM URL
# ============================================================
# Reads an image from URL, converts it to RGB and grayscale,
# prints its dimensions, and displays the RGB image.

url = 'https://fatcatart.com/wp-content/uploads/2019/03/Van-Gogh-Starry-Night-cat-w.jpg'
response = requests.get(url)
PIL_img = Image.open(BytesIO(response.content)).convert('RGB')
PIL_img_gray = PIL_img.convert('L')

print('Image resolution:', PIL_img.size)
img_grid([PIL_img], 'Original Image', ['RGB Image'], cols=1, figsize=(8, 6))


# ============================================================
# SECTION 4 - READ LOCAL IMAGE USING OPENCV
# ============================================================
# Upload an image in Colab first, then set image_path.
# OpenCV reads color images as BGR, so convert BGR -> RGB
# before displaying via Matplotlib.

# from google.colab import files
# uploaded = files.upload()
# image_path = list(uploaded.keys())[0]
#
# image_bgr = cv2.imread(image_path)
# if image_bgr is None:
#     print('Error importing image')
# else:
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     print('OpenCV image shape:', image_bgr.shape)
#     plt.imshow(image_rgb)
#     plt.axis('off')
#     plt.title('OpenCV BGR Converted to RGB')
#     plt.show()


# ============================================================
# SECTION 5 - SHOW GRAYSCALE IMAGE
# ============================================================
# Converts RGB image to grayscale. Pixel intensities are 0-255.

gray_np = np.array(PIL_img_gray)
img_grid([PIL_img, PIL_img_gray], 'Original and Grayscale', ['Original RGB', 'Grayscale'], cols=2)


# ============================================================
# SECTION 6 - NEGATIVE IMAGE
# ============================================================
# Creates RGB and grayscale negative images.

negative_rgb = ImageOps.invert(PIL_img)
negative_gray = ImageOps.invert(PIL_img_gray)
img_grid(
    [PIL_img, negative_rgb, PIL_img_gray, negative_gray],
    'Negative Images',
    ['Original RGB', 'RGB Negative', 'Original Grayscale', 'Grayscale Negative'],
    cols=4
)


# ============================================================
# SECTION 7 - RGB COLOR CHANNELS
# ============================================================
# Splits image into red, green and blue channel images.

r, g, b = PIL_img.split()
img_grid([PIL_img, r, g, b], 'RGB Channels', ['Original', 'Red', 'Green', 'Blue'], cols=4)


# ============================================================
# SECTION 8 - COLORED RGB CHANNELS
# ============================================================
# Displays each individual RGB channel in its own actual color.

def colored_channels(image):
    channels = image.split()
    shape = np.array(image).shape
    result = []
    for i in range(3):
        output = np.zeros(shape, dtype=np.uint8)
        output[:, :, i] = np.array(channels[i])
        result.append(Image.fromarray(output, 'RGB'))
    return result

r_color, g_color, b_color = colored_channels(PIL_img)
img_grid(
    [PIL_img, PIL_img_gray, negative_gray, r_color, g_color, b_color],
    'Original, Grayscale, Negative, and Colored Channels',
    ['Original', 'Grayscale', 'Gray Negative', 'Red Only', 'Green Only', 'Blue Only'],
    cols=3
)


# ============================================================
# SECTION 9 - SPATIAL RESOLUTION REDUCTION
# ============================================================
# Reduces image size to 128x128 using bicubic interpolation.

low_res = PIL_img.resize((128, 128), Image.BICUBIC)
print('Low-resolution size:', low_res.size)
img_grid([PIL_img, low_res], 'Spatial Resolution', ['Original', '128 x 128'], cols=2)


# ============================================================
# SECTION 10 - INTENSITY RESOLUTION: 8 BITS TO 3 BITS
# ============================================================
# 3 bits means 2^3 = 8 intensity levels.
# To reduce from 8 bits to 3 bits, shift/divide by 2^5.

bit_reduction = 5
gray_8bit = np.array(PIL_img.convert('L'))
reduced = gray_8bit // (2 ** bit_reduction)
rescaled = (reduced * (255 // 7)).astype(np.uint8)

print('Reduced maximum:', reduced.max())
print('Reduced minimum:', reduced.min())
img_grid([gray_8bit, rescaled], 'Intensity Resolution: 8-bit to 3-bit', ['Original 8-bit', 'Reduced 3-bit'], cols=2)


# ============================================================
# SECTION 11 - HISTOGRAM AND HEATMAP
# ============================================================
# Histogram: brightness distribution.
# Heatmap: actual intensity values in a small 15x15 image.

small_gray = cv2.resize(gray_8bit, (15, 15))
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(gray_8bit.flatten(), bins=30, kde=True, color='blue', ax=axes[0])
axes[0].set_title('Image Histogram')
axes[0].set_xlabel('Pixel Intensity (0 = Black, 255 = White)')
sns.heatmap(small_gray, annot=True, fmt='d', cmap='gray', cbar=False, ax=axes[1])
axes[1].set_title('15 x 15 Pixel Intensity Heatmap')
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 12 - BASIC IMAGE OPERATIONS
# ============================================================
# Rotate an image, adjust brightness with clipping, and subtract images.

coins = ski.data.coins()
coins_rotated = (ski.transform.rotate(coins, 2) * 255).astype(np.uint8)
coins_bright = np.clip(coins_rotated.astype(np.int16) + 50, 0, 255).astype(np.uint8)
difference = np.clip(coins.astype(np.int16) - coins_rotated.astype(np.int16), 0, 255).astype(np.uint8)

img_grid([coins, coins_rotated, coins_bright, difference],
         'Basic Image Operations',
         ['Original', 'Rotated', 'Brightness +50', 'Original - Rotated'], cols=4)


# ============================================================
# SECTION 13 - BIT-PLANE SLICING
# ============================================================
# Extracts all 8 bit planes from a grayscale image.

def bitplane_slice(image):
    planes = []
    for i in range(8):
        plane = (image & (1 << i))
        plane = (plane > 0).astype(np.uint8) * 255
        planes.append(plane)
    return planes

bitplanes = bitplane_slice(gray_8bit)
img_grid([gray_8bit] + bitplanes,
         'Bit-Plane Slicing',
         ['Original'] + [f'Bitplane {i}' for i in range(8)], cols=3)


# ============================================================
# SECTION 14 - BIT-PLANE RECONSTRUCTION
# ============================================================
# Reconstructs an image from selected bit planes.

def reconstruct_from_bitplanes(planes, indices):
    output = np.zeros_like(planes[0], dtype=np.int64)
    for i in indices:
        output += (planes[i] // 255) * (2 ** i)
    return np.clip(output, 0, 255).astype(np.uint8)

reconstructed_high4 = reconstruct_from_bitplanes(bitplanes, [7, 6, 5, 4])
reconstructed_high2 = reconstruct_from_bitplanes(bitplanes, [7, 6])
img_grid([gray_8bit, reconstructed_high4, reconstructed_high2],
         'Bit-Plane Reconstruction',
         ['Original', 'Planes 7,6,5,4', 'Planes 7,6'], cols=3)


# ============================================================
# SECTION 15 - RECTANGULAR MASKING
# ============================================================
# Creates a mask that keeps only the right half of an RGB image.

rgb_np = np.array(PIL_img)
h, w, c = rgb_np.shape
right_mask = np.zeros((h, w, 3), dtype=np.uint8)
right_mask[:, w // 2:, :] = 1
right_output = rgb_np * right_mask
img_grid([rgb_np, right_mask * 255, right_output],
         'Rectangular Masking', ['Original', 'Right-Half Mask', 'Output'], cols=3)


# ============================================================
# SECTION 16 - CIRCULAR MASKING
# ============================================================
# Creates a circular mask. This version keeps pixels OUTSIDE circle.
# Use <= instead of >= to keep pixels INSIDE circle.

circle_mask = np.zeros((h, w, 3), dtype=np.uint8)
cx, cy = w // 2, h // 2
radius = 100
Y, X = np.ogrid[:h, :w]
outside_circle = (X - cx) ** 2 + (Y - cy) ** 2 >= radius ** 2
circle_mask[outside_circle] = 1
circle_output = rgb_np * circle_mask
img_grid([rgb_np, circle_mask * 255, circle_output],
         'Circular Masking', ['Original', 'Outside-Circle Mask', 'Output'], cols=3)


# ============================================================
# SECTION 17 - LOG TRANSFORMATION
# ============================================================
# Enhances dark regions using s = c * log(1 + r).

def log_transform(image, c=25):
    output = c * np.log(1 + image.astype(np.float32))
    return np.clip(output, 0, 255).astype(np.uint8)

log_image = log_transform(np.array(PIL_img))
img_grid([PIL_img, log_image], 'Log Transformation', ['Original', 'Log Transformed'], cols=2)


# ============================================================
# SECTION 18 - GAMMA TRANSFORMATION
# ============================================================
# Gamma < 1 brightens; gamma > 1 darkens.

def gamma_transform(image, gamma, c=1):
    normalized = image.astype(np.float32) / 255.0
    output = c * (normalized ** gamma)
    return np.clip(255 * output, 0, 255).astype(np.uint8)

gamma_images = [
    np.array(PIL_img),
    gamma_transform(np.array(PIL_img), 0.5),
    gamma_transform(np.array(PIL_img), 0.2),
    gamma_transform(np.array(PIL_img), 0.02),
    gamma_transform(np.array(PIL_img), 1.5),
    gamma_transform(np.array(PIL_img), 2.5),
    gamma_transform(np.array(PIL_img), 10)
]
img_grid(gamma_images, 'Gamma Transformation',
         ['Original', '0.5', '0.2', '0.02', '1.5', '2.5', '10'], cols=4)


# ============================================================
# SECTION 19 - HISTOGRAM STRETCHING
# ============================================================
# Maps original min/max intensity to the full 0-255 range.

def histogram_stretching(image, min_val=0, max_val=255):
    old_min = image.min()
    old_max = image.max()
    stretched = ((image - old_min) * ((max_val - min_val) / (old_max - old_min))) + min_val
    return np.clip(stretched, min_val, max_val).astype(np.uint8)

log_gray = log_transform(gray_8bit)
subtracted = np.clip(log_gray.astype(np.int16) - 100, 0, 255).astype(np.uint8)
stretched = histogram_stretching(subtracted)
img_grid([subtracted, stretched], 'Histogram Stretching', ['Input', 'Stretched'], cols=2)


# ============================================================
# SECTION 20 - SINGLE THRESHOLDING (k = 128)
# ============================================================
# Creates a binary image and multiplies original grayscale by mask.

k = 128
binary_mask = gray_8bit < k
binary_display = binary_mask.astype(np.uint8) * 255
threshold_output = gray_8bit * binary_mask.astype(np.uint8)
img_grid([gray_8bit, binary_display, threshold_output],
         'Single Thresholding (k = 128)',
         ['Original', 'Binary Mask', 'Original x Mask'], cols=3)


# ============================================================
# SECTION 21 - TWO-SIDED THRESHOLDING + INVERSE MASK
# ============================================================
# Keeps only intensities between lower_threshold and upper_threshold.

lower_threshold = 50
upper_threshold = 210
two_sided_mask = np.zeros_like(gray_8bit)
two_sided_mask[(gray_8bit > lower_threshold) & (gray_8bit < upper_threshold)] = 255
mask_rgb = cv2.cvtColor(two_sided_mask, cv2.COLOR_GRAY2RGB)
masked_rgb = (np.array(PIL_img) * (mask_rgb / 255.0)).astype(np.uint8)
inverse_mask_rgb = skimage.util.invert(mask_rgb) / 255.0
inverse_output = (np.array(PIL_img) * inverse_mask_rgb).astype(np.uint8)

img_grid([PIL_img, mask_rgb, masked_rgb, inverse_output],
         'Two-Sided Thresholding',
         ['Original', 'Two-Sided Mask', 'Original x Mask', 'Original x Inverse Mask'], cols=4)


# ============================================================
# SECTION 22 - HISTOGRAM EQUALIZATION
# ============================================================
# Improves global contrast by redistributing grayscale intensities.

def histogram_equalization(image):
    return (exposure.equalize_hist(image) * 255).astype(np.uint8)

equalized = histogram_equalization(gray_8bit)
img_grid([gray_8bit, equalized], 'Histogram Equalization', ['Original', 'Equalized'], cols=2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(gray_8bit.flatten(), bins=256, color='gray')
plt.title('Original Histogram')
plt.subplot(1, 2, 2)
sns.histplot(equalized.flatten(), bins=256, color='gray')
plt.title('Equalized Histogram')
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 23 - HISTOGRAM MATCHING
# ============================================================
# Matches the source image histogram to a reference image histogram.

reference_url = 'https://media.geeksforgeeks.org/wp-content/uploads/20190721215512/sample.jpg'
reference = Image.open(BytesIO(requests.get(reference_url).content)).convert('L')
reference_np = np.array(reference)
matched = match_histograms(gray_8bit, reference_np)
matched = np.clip(matched, 0, 255).astype(np.uint8)
img_grid([gray_8bit, reference_np, matched],
         'Histogram Matching', ['Source', 'Reference', 'Matched'], cols=3)


# ============================================================
# SECTION 24 - MEAN FILTER / AVERAGE BLUR
# ============================================================
# Convolution with an n x n averaging kernel blurs image.

n = 9
mean_kernel = np.ones((n, n), dtype=np.float32) / (n * n)
mean_filtered = convolve(gray_8bit, mean_kernel)
mean_filtered = np.clip(mean_filtered, 0, 255).astype(np.uint8)
img_grid([gray_8bit, mean_filtered], 'Mean Filtering', ['Original', f'Mean Filter {n}x{n}'], cols=2)


# ============================================================
# SECTION 25 - ADD SALT-PEPPER AND GAUSSIAN NOISE
# ============================================================
# Noise helper functions for denoising and PSNR questions.

def add_salt_and_pepper_noise(image, prob):
    normalized = image / 255.0
    noisy = random_noise(normalized, mode='s&p', amount=prob)
    return (noisy * 255).astype(np.uint8)

def add_gaussian_noise(image, mean=0, var=0.1):
    normalized = image / 255.0
    noisy = random_noise(normalized, mode='gaussian', mean=mean, var=var)
    return (noisy * 255).astype(np.uint8)

sp_low = add_salt_and_pepper_noise(gray_8bit, 0.02)
sp_medium = add_salt_and_pepper_noise(gray_8bit, 0.10)
sp_high = add_salt_and_pepper_noise(gray_8bit, 0.50)
gauss_low = add_gaussian_noise(gray_8bit, var=0.01)
gauss_medium = add_gaussian_noise(gray_8bit, var=0.05)
gauss_high = add_gaussian_noise(gray_8bit, var=0.50)

img_grid([gray_8bit, sp_low, sp_medium, sp_high], 'Salt-and-Pepper Noise',
         ['Original', 'Low', 'Medium', 'High'], cols=4)
img_grid([gray_8bit, gauss_low, gauss_medium, gauss_high], 'Gaussian Noise',
         ['Original', 'Low', 'Medium', 'High'], cols=4)


# ============================================================
# SECTION 26 - MEDIAN AND GAUSSIAN DENOISING
# ============================================================
# Median filter removes salt-and-pepper noise.
# Gaussian blur reduces Gaussian noise.

def apply_median_filter(image, size=3):
    return median_filter(image, size=size)

def apply_gaussian_filter(image, ksize=5, sigma=5):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

sp_low_denoised = apply_median_filter(sp_low, 3)
gauss_low_denoised = apply_gaussian_filter(gauss_low, 5, 5)

img_grid([sp_low, sp_low_denoised, gauss_low, gauss_low_denoised],
         'Noise Removal',
         ['Salt & Pepper', 'Median Filtered', 'Gaussian Noise', 'Gaussian Filtered'], cols=2)


# ============================================================
# SECTION 27 - PREWITT EDGE DETECTION
# ============================================================
# Uses horizontal and vertical Prewitt-like convolution kernels.

prewitt_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
prewitt_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
edge_input = gray_8bit.astype(np.int16)
horizontal_edges = np.clip(convolve(edge_input, prewitt_h), 0, 255).astype(np.uint8)
vertical_edges = np.clip(convolve(edge_input, prewitt_v), 0, 255).astype(np.uint8)

img_grid([gray_8bit, horizontal_edges, vertical_edges],
         'Prewitt Edge Detection', ['Original', 'Horizontal Edges', 'Vertical Edges'], cols=3)


# ============================================================
# SECTION 28 - LAPLACIAN FILTER AND SHARPENING
# ============================================================
# Applies Laplacian kernels and produces sharpened outputs.

laplace_kernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplace_kernel2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
laplace_input = gray_8bit.astype(np.int16)
laplace_raw1 = convolve(laplace_input, laplace_kernel1)
laplace_raw2 = convolve(laplace_input, laplace_kernel2)
laplace1 = np.clip(laplace_raw1, 0, 255).astype(np.uint8)
laplace2 = np.clip(laplace_raw2, 0, 255).astype(np.uint8)
sharpen1 = np.clip(laplace_input - laplace_raw1, 0, 255).astype(np.uint8)
sharpen2 = np.clip(laplace_input + laplace_raw2, 0, 255).astype(np.uint8)

img_grid([gray_8bit, laplace1, laplace2, sharpen1, sharpen2],
         'Laplacian Filters and Sharpening',
         ['Original', 'Laplacian 1', 'Laplacian 2', 'Sharpened 1', 'Sharpened 2'], cols=3)


# ============================================================
# SECTION 29 - UNSHARP MASKING
# ============================================================
# Unsharp masking enhances details and edges.

from skimage.filters import unsharp_mask
unsharp1 = unsharp_mask(gray_8bit, radius=1)
unsharp5 = unsharp_mask(gray_8bit, radius=5)
unsharp10 = unsharp_mask(gray_8bit, radius=10)
img_grid([gray_8bit, unsharp1, unsharp5, unsharp10],
         'Unsharp Masking', ['Original', 'Radius 1', 'Radius 5', 'Radius 10'], cols=4)


# ============================================================
# SECTION 30 - PSNR
# ============================================================
# Higher PSNR means test image is closer to ground truth.

def PSNR(ground_truth, test_image):
    return skimage.metrics.peak_signal_noise_ratio(ground_truth, test_image)

print('PSNR original vs original:', PSNR(gray_8bit, gray_8bit))
print('PSNR original vs salt-pepper high:', PSNR(gray_8bit, sp_high))
print('PSNR original vs Gaussian low:', PSNR(gray_8bit, gauss_low))


# ============================================================
# SECTION 31 - DFT / FFT AND INVERSE FFT
# ============================================================
# Computes FFT, shifts spectrum to center, restores shift,
# and reconstructs image using inverse FFT.

fft_input = gray_8bit
fft = np.fft.fft2(fft_input)
fft_shifted = np.fft.fftshift(fft)
fft_unshifted = np.fft.ifftshift(fft_shifted)
ifft = np.fft.ifft2(fft_unshifted)

img_grid([
    fft_input,
    np.log(np.abs(fft) + 1),
    np.log(np.abs(fft_shifted) + 1),
    np.log(np.abs(fft_unshifted) + 1),
    np.abs(ifft)
], 'DFT / FFT Transform',
['Original', 'FFT Magnitude', 'Shifted FFT', 'Inverse Shift', 'Reconstructed'], cols=5, figsize=(22, 6))


# ============================================================
# SECTION 32 - IDEAL LOW-PASS FILTER (ILPF)
# ============================================================
# Keeps only low frequencies inside a circular radius.

def ideal_low_pass_mask(shape, radius):
    height, width = shape
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    return (((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2).astype(np.uint8)

radius = 50
ilpf_mask = ideal_low_pass_mask(fft_shifted.shape, radius)
fft_low = fft_shifted * ilpf_mask
ilpf_output = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_low)))

img_grid([fft_input, ilpf_mask, np.log(np.abs(fft_low) + 1), ilpf_output],
         'Ideal Low-Pass Filter',
         ['Original', f'ILPF Mask r={radius}', 'Filtered Spectrum', 'Output'], cols=4)
print('PSNR ILPF:', PSNR(fft_input, ilpf_output))


# ============================================================
# SECTION 33 - IDEAL HIGH-PASS FILTER (IHPF)
# ============================================================
# Inverts low-pass mask. Keeps high frequencies and highlights details.

ihpf_mask = 1 - ilpf_mask
fft_high = fft_shifted * ihpf_mask
ihpf_output = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_high)))

img_grid([fft_input, ihpf_mask, np.log(np.abs(fft_high) + 1), ihpf_output],
         'Ideal High-Pass Filter',
         ['Original', f'IHPF Mask r={radius}', 'Filtered Spectrum', 'Output'], cols=4)


# ============================================================
# SECTION 34 - LOW-PASS FILTER WITH MULTIPLE RADII
# ============================================================
# Compare blur level for different ideal low-pass radii.

def ideal_low_pass_filter(image, radius):
    f = np.fft.fft2(image)
    shifted = np.fft.fftshift(f)
    mask = ideal_low_pass_mask(shifted.shape, radius)
    return np.abs(np.fft.ifft2(np.fft.ifftshift(shifted * mask)))

lp10 = ideal_low_pass_filter(fft_input, 10)
lp20 = ideal_low_pass_filter(fft_input, 20)
lp50 = ideal_low_pass_filter(fft_input, 50)
img_grid([lp10, lp20, lp50], 'ILPF Radius Comparison', ['Radius 10', 'Radius 20', 'Radius 50'], cols=3)


# ============================================================
# SECTION 35 - MULTIPLY FILTERED LOW-PASS RESULTS
# ============================================================
# Multiplies outputs of low-pass filters and normalizes for display.

combined = lp10 * lp20 * lp50
combined = ((combined - combined.min()) / (combined.max() - combined.min()) * 255).astype(np.uint8)
img_grid([combined], 'Multiplied Low-Pass Results', ['R10 x R20 x R50'], cols=1, figsize=(8, 8))


# ============================================================
# SECTION 36 - BUTTERWORTH LOW-PASS FILTER (BLPF)
# ============================================================
# Smooth frequency cutoff. D0 is cutoff radius; n is order.

def butterworth_low_pass_mask(shape, D0, n=2):
    height, width = shape
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return 1 / (1 + (distance / D0) ** (2 * n))

D0 = 50
order = 2
blpf_mask = butterworth_low_pass_mask(fft_shifted.shape, D0, order)
blpf_output = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shifted * blpf_mask)))
img_grid([fft_input, blpf_mask, np.log(np.abs(fft_shifted * blpf_mask) + 1), blpf_output],
         'Butterworth Low-Pass Filter', ['Original', 'BLPF Mask', 'Filtered Spectrum', 'Output'], cols=4)


# ============================================================
# SECTION 37 - BUTTERWORTH HIGH-PASS FILTER (BHPF)
# ============================================================
# High-pass mask equals 1 - Butterworth low-pass mask.

bhpf_mask = 1 - blpf_mask
bhpf_output = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shifted * bhpf_mask)))
img_grid([fft_input, bhpf_mask, np.log(np.abs(fft_shifted * bhpf_mask) + 1), bhpf_output],
         'Butterworth High-Pass Filter', ['Original', 'BHPF Mask', 'Filtered Spectrum', 'Output'], cols=4)


# ============================================================
# SECTION 38 - GAUSSIAN LOW-PASS FILTER (GLPF)
# ============================================================
# Gaussian frequency filter with smooth transition.

def gaussian_low_pass_mask(shape, D0):
    height, width = shape
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    distance_squared = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-distance_squared / (2 * D0 ** 2))

glpf_mask = gaussian_low_pass_mask(fft_shifted.shape, D0=50)
glpf_output = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shifted * glpf_mask)))
img_grid([fft_input, glpf_mask, np.log(np.abs(fft_shifted * glpf_mask) + 1), glpf_output],
         'Gaussian Low-Pass Filter', ['Original', 'GLPF Mask', 'Filtered Spectrum', 'Output'], cols=4)


# ============================================================
# SECTION 39 - GAUSSIAN HIGH-PASS FILTER (GHPF)
# ============================================================
# Inverts the Gaussian low-pass mask.

ghpf_mask = 1 - glpf_mask
ghpf_output = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shifted * ghpf_mask)))
img_grid([fft_input, ghpf_mask, np.log(np.abs(fft_shifted * ghpf_mask) + 1), ghpf_output],
         'Gaussian High-Pass Filter', ['Original', 'GHPF Mask', 'Filtered Spectrum', 'Output'], cols=4)


# ============================================================
# SECTION 40 - DCT AND INVERSE DCT
# ============================================================
# Computes 2D DCT and reconstructs image with inverse DCT.

def dct2(image):
    return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(coefficients):
    return idct(idct(coefficients, axis=0, norm='ortho'), axis=1, norm='ortho')

dct_image = dct2(fft_input)
dct_output = np.abs(idct2(dct_image))
img_grid([fft_input, np.log(np.abs(dct_image) + 1), dct_output],
         'DCT Transform and Reconstruction', ['Original', 'DCT Magnitude', 'Reconstructed'], cols=3)
print('PSNR DCT reconstruction:', PSNR(fft_input, dct_output))


# ============================================================
# SECTION 41 - FREQUENCY DOMAIN COMPRESSION
# ============================================================
# Keeps only coefficients above a selected magnitude percentile.

compression_fft = np.fft.fft2(fft_input)
compression_shifted = np.fft.fftshift(compression_fft)
compression_magnitude = np.abs(compression_shifted)
percentile = 75
threshold = np.percentile(compression_magnitude, percentile)
compression_mask = compression_magnitude >= threshold
compression_filtered = compression_shifted * compression_mask.astype(int)
compression_output = np.abs(np.fft.ifft2(np.fft.ifftshift(compression_filtered)))
compression_output_uint8 = np.clip(compression_output, 0, 255).astype(np.uint8)

img_grid([fft_input, np.log(compression_magnitude + 1), np.log(np.abs(compression_filtered) + 1), compression_output_uint8],
         'Frequency Domain Compression', ['Original', 'FFT', 'Thresholded FFT', 'Output'], cols=4)
print('PSNR frequency compression:', PSNR(fft_input, compression_output_uint8))


# ============================================================
# SECTION 42 - EROSION AND DILATION
# ============================================================
# Erosion shrinks bright regions; dilation expands bright regions.

morph_input = gray_8bit
kernel_size = 15
morph_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
eroded = erosion(morph_input, footprint=morph_kernel)
dilated = dilation(morph_input, footprint=morph_kernel)
img_grid([morph_input, eroded, dilated], 'Morphological Operations', ['Original', 'Erosion', 'Dilation'], cols=3)


# ============================================================
# SECTION 43 - GLOBAL, OTSU, AND ADAPTIVE THRESHOLDING
# ============================================================
# Global: fixed threshold.
# Otsu: automatic global threshold.
# Adaptive: local threshold, useful for uneven lighting.

from skimage.filters import threshold_otsu, threshold_local
page_image = data.page()
global_threshold = 128
global_binary = page_image > global_threshold
otsu_threshold = threshold_otsu(page_image)
otsu_binary = page_image > otsu_threshold
local_threshold = threshold_local(page_image, block_size=35, offset=10)
adaptive_binary = page_image > local_threshold

img_grid([page_image, global_binary * 255, otsu_binary * 255, adaptive_binary * 255],
         'Thresholding Methods', ['Original', 'Global', f'Otsu ({otsu_threshold})', 'Adaptive'], cols=4)


# ============================================================
# SECTION 44 - WATERSHED SEGMENTATION
# ============================================================
# Uses Sobel elevation map and foreground/background markers.

coins_image = data.coins()
elevation = filters.sobel(coins_image)
markers = np.zeros_like(coins_image)
markers[coins_image < 30] = 1
markers[coins_image > 150] = 2
watershed_result = segmentation.watershed(elevation, markers)

img_grid([coins_image, elevation, markers * 127, watershed_result * 127],
         'Watershed Segmentation', ['Original', 'Sobel Elevation', 'Markers', 'Watershed'], cols=4)


# ============================================================
# SECTION 45 - CONNECTED COMPONENT LABELING
# ============================================================
# Labels each separate binary foreground object with an integer ID.

coins_otsu = threshold_otsu(coins_image)
coins_binary = coins_image > coins_otsu
labeled_coins, count = label(coins_binary)
print('Connected components:', count)
img_grid([coins_binary * 255, labeled_coins], 'Connected Components', ['Binary Image', 'Labeled Objects'], cols=2)


# ============================================================
# SECTION 46 - LABEL COLOR OVERLAY
# ============================================================
# Displays labelled connected components in different colors.

overlay = label2rgb(labeled_coins, image=coins_image, bg_label=0)
overlay_uint8 = (overlay * 255).astype(np.uint8)
img_grid([overlay_uint8], 'Label Overlay', ['Colored Components'], cols=1, figsize=(9, 8))


# ============================================================
# SECTION 47 - HOUGH CIRCLE TRANSFORM
# ============================================================
# Detects circular objects from Canny edges using possible radii.

hough_image = data.coins()
edges = canny(hough_image, sigma=2, low_threshold=10, high_threshold=50)
hough_radii = np.arange(10, 60, 2)
hough_result = hough_circle(edges, hough_radii)
accums, cx, cy, radii = hough_circle_peaks(hough_result, hough_radii, total_num_peaks=30)

circle_output = color.gray2rgb(hough_image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circle_y, circle_x = circle_perimeter(center_y, center_x, radius, shape=circle_output.shape)
    circle_output[circle_y, circle_x] = (220, 20, 20)

img_grid([hough_image, edges * 255, circle_output],
         'Hough Circle Transform', ['Original Coins', 'Canny Edges', 'Detected Circles'], cols=3)


# ============================================================
# SECTION 48 - OPTIONAL HOUGH LINE TRANSFORM
# ============================================================
# Uncomment this code only if the exam asks for line detection.

# from skimage.transform import probabilistic_hough_line
# line_image = data.camera()
# line_edges = canny(line_image, sigma=2)
# lines = probabilistic_hough_line(line_edges, threshold=10, line_length=5, line_gap=3)
# plt.figure(figsize=(8, 8))
# plt.imshow(line_image, cmap='gray')
# for p0, p1 in lines:
#     plt.plot((p0[0], p1[0]), (p0[1], p1[1]), 'r-')
# plt.axis('off')
# plt.title('Hough Line Detection')
# plt.show()


# ============================================================
# END OF FINAL REFERENCE
# ============================================================
# Quick exam index:
# 1-11: Image loading, channels, resize, intensity, histogram
# 12-19: Operations, bit planes, masking, log/gamma, stretching
# 20-23: Thresholding, equalization, histogram matching
# 24-29: Filters, noise, denoising, edges, sharpening, PSNR
# 30-41: DFT, DCT, compression, ideal/Butterworth/Gaussian filters
# 42-48: Morphology, thresholding, watershed, labels, Hough transforms
