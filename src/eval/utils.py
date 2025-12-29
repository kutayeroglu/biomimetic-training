"""
Utility functions for evaluation, including metric calculations and image processing.
"""
import os
import math
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image


def normalize(x, x_min=None, x_max=None):
    """Normalize a value between a minimum and maximum value."""
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)


def calc_color_index(img, n=0):
    """Calculate the color index of an image."""
    x = img[:, :, 0] * math.cos(0) + img[:, :, 1] * math.cos(math.radians(120)) + img[:, :, 2] * math.cos(math.radians(-120))
    y = img[:, :, 0] * math.sin(0) + img[:, :, 1] * math.sin(math.radians(120)) + img[:, :, 2] * math.sin(math.radians(-120))
    col_vec_length = np.sqrt(x * x + y * y)
    
    if n == 0:
        return np.mean(col_vec_length, axis=None)
    else:
        return np.mean(np.flip(np.sort(col_vec_length, axis=None))[0:n])


def rgb2gray(rgb):
    """Convert an RGB image to grayscale."""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def make_larger(original, n_times):
    """Make an image larger by a factor of n_times."""
    large = np.zeros((original.shape[0]*n_times, original.shape[1]*n_times))
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            large[i * n_times:(i + 1)*n_times, j * n_times:(j + 1) * n_times] = original[i,j]
    return large


def radial_average(ps):
    """Calculate the radial average of an image."""
    h = ps.shape[0]
    w = ps.shape[1]
    h_mid = h // 2
    w_mid = w // 2
    r_max = np.min((w_mid, h_mid))
    
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(X - w_mid, Y - h_mid).astype(int)
    r_index = np.arange(0, r_max)
    means = ndimage.mean(ps, r, index=r_index)
    return r_index, means


def azimuthal_average(ps, angle_interval, r_min, r_max):
    """Calculate the azimuthal average of an image."""
    h = ps.shape[0]
    w = ps.shape[1]
    h_mid = h // 2
    w_mid = w // 2
    
    Y, X = np.ogrid[0:h, 0:w]
    r = np.hypot(-(Y - h_mid), (X - w_mid))
    mask = np.logical_and(r>r_min, r<r_max)
    
    theta = np.rad2deg(np.arctan2(-(Y - h_mid), (X - w_mid)))
    theta = np.mod(theta + angle_interval / 2 + 360, 360)
    theta = (angle_interval * (theta // angle_interval)).astype(int)
    dia = np.multiply(np.ones(theta.shape) * 180, (theta >= 180).astype(int))
    theta = theta - dia + 1
    theta = np.multiply(mask, theta)
    theta = theta - 1
    
    angle_index = np.arange(0, 180, int(angle_interval))
    means = ndimage.mean(ps, theta, index=angle_index)
    return angle_index, means


def fft_average(abs_spectrum, n_times, angle_interval):
    """Calculate the radial and azimuthal averages of the FFT of an image."""
    as_large = make_larger(abs_spectrum, n_times)
    r_index, r_means = radial_average(as_large)
    a_index, a_means = azimuthal_average(as_large, angle_interval, 0, abs_spectrum.shape[0] * n_times / 4)
    return r_index, r_means, a_index, a_means


def mean_resultant_length(a_index, a_mean):
    """Calculate the mean resultant length of the azimuthal average."""
    return np.abs(np.sum(a_mean * np.exp(2 * math.pi * 1j * a_index / 180))) / np.sum(a_mean)


def calc_fft_index(img, n_times, angle_interval, return_raw=False):
    """Calculate the frequency index of the FFT of an image."""
    if len(img.shape) == 3:
        img = rgb2gray(img)
    gray_fft = np.fft.fftshift(np.fft.fft2(img))
    abs_spectrum = np.abs(gray_fft)
    half_size = img.shape[0] // 2
    if img.shape[0] % 2 == 0:
        abs_spectrum[half_size - 1:half_size + 1, half_size - 1:half_size + 1] = 0
    else:
        abs_spectrum[half_size, half_size] = 0
    r_index, r_means, a_index, a_means = fft_average(abs_spectrum, n_times, angle_interval)
    
    # weighted average of frequency
    r_means = r_means / np.sum(r_means)
    peak_freq = np.sum(r_means*r_index)/np.sum(r_means)
    peak_freq = peak_freq / n_times
    
    # mean resultant length
    mrl = mean_resultant_length(a_index, a_means)
    
    if return_raw:
        return peak_freq, mrl, r_index, r_means, a_index, a_means
    return peak_freq, mrl


def calc_rf_indices(weights, n_top_col_pixel=48, n_times=100, angle_interval=1, return_rank=False, color_only=False):
    """
    Calculate the indices of the receptor fields (first conv layer weights).
    Note: PyTorch weights are [out_channels, in_channels, H, W]
    TensorFlow weights are [H, W, in_channels, out_channels]
    We transpose to TensorFlow format for processing, then convert back.
    """
    assert n_top_col_pixel >= 0, "n_top_col_pixel must be greater than or equal to 0"
    assert n_times > 0, "n_times must be greater than 0"
    assert angle_interval > 0, "angle_interval must be greater than 0"
    assert isinstance(return_rank, bool), "return_rank must be a boolean"
    assert len(weights.shape) == 4, "Weights must be 4D"
    
    # PyTorch: [out_channels, in_channels, H, W] -> transpose to [H, W, in_channels, out_channels] for processing
    weights_tf_format = np.transpose(weights, (2, 3, 1, 0))  # [H, W, in_channels, out_channels]
    
    # Check if weights are square (H == W) for the assertion
    assert weights_tf_format.shape[0] == weights_tf_format.shape[1], "Weights must be square in spatial dimensions"
    
    weights_norm = np.zeros(weights_tf_format.shape)
    num_rf = weights_tf_format.shape[3]  # out_channels
    for i in range(num_rf):
        weights_norm[:,:,:,i] = normalize(weights_tf_format[:,:,:,i])
    
    # Color index
    color_index = np.array([calc_color_index(weights_norm[:,:,:,i], n_top_col_pixel) for i in range(num_rf)])
    
    if color_only:
        if return_rank:
            return np.argsort(color_index)
        else:
            return color_index
    
    # FFT index
    fft_freq_index = []
    fft_az_index = []
    for i in range(num_rf):
        peak_freq, mrl = calc_fft_index(weights_norm[:,:,:,i], n_times, angle_interval)
        fft_freq_index.append(peak_freq)
        fft_az_index.append(mrl)
    
    fft_freq_index = np.array(fft_freq_index)
    fft_az_index = np.array(fft_az_index)
    
    if return_rank:
        color_rank = np.argsort(color_index)
        fft_freq_rank = np.argsort(fft_freq_index)
        fft_az_rank = np.argsort(fft_az_index)
        return color_rank, fft_freq_rank, fft_az_rank
    
    return color_index, fft_freq_index, fft_az_index


def load_images(list_images, img_size):
    """Load images from a list of file paths and return the images and labels."""
    input_images = np.zeros(shape=(len(list_images), *img_size), dtype=np.uint8)
    class_labels = []
    
    for i, file in enumerate(list_images):
        newimg = Image.open(file)
        if newimg.mode != 'RGB':
            newimg = newimg.convert('RGB')
        newimg = np.asarray(newimg).astype(np.uint8)
        class_labels.append(os.path.basename(file).split('.')[0])
        input_images[i] = newimg
    
    return input_images, class_labels


def make_decision(probability, cate16_class_indices):
    """Maps 1000-class probabilities to 16 entry-level categories."""
    max_value = -1
    category_decision = None
    for category in cate16_class_indices.keys():
        indices = cate16_class_indices[category]
        values = np.take(probability, indices)
        aggregated_value = np.max(values)
        if aggregated_value > max_value:
            max_value = aggregated_value
            category_decision = category
    return category_decision

