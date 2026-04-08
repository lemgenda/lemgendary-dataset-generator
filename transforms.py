import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter

def add_gaussian_noise(img, sigma_range=(5, 50)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_poisson_noise(img):
    img_float = img.astype(np.float32) / 255.0
    vals = len(np.unique(img_float))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img_float * vals) / float(vals)
    return (np.clip(noisy, 0, 1) * 255).astype(np.uint8)

def add_motion_blur(img, kernel_size=(15, 30)):
    size = random.randint(*kernel_size)
    kernel_v = np.zeros((size, size))
    angle = random.uniform(0, 360)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel_v[int((size - 1) / 2), :] = np.ones(size)
    kernel_v = cv2.warpAffine(kernel_v, matrix, (size, size))
    kernel_v = kernel_v / np.sum(kernel_v)
    return cv2.filter2D(img, -1, kernel_v)

def add_gaussian_blur(img, kernel_range=(3, 11)):
    ksize = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_jpeg_compression(img, quality_range=(30, 90)):
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def low_light_transform(img):
    # Simulate low light by architectural depression and gamma shifting
    gamma = random.uniform(1.8, 3.5)
    brightness_scale = random.uniform(0.1, 0.4)
    img_float = img.astype(np.float32) / 255.0
    img_low = np.power(img_float * brightness_scale, gamma)
    return (np.clip(img_low, 0, 1) * 255).astype(np.uint8)

def degrade(img, task):
    """
    Apply task-specific synthetic degradations.
    Returns: (degraded_img, metadata)
    """
    meta = {"original_task": task}
    
    if task == "nafnet":
        # Deblurring + Denoising focus
        if random.random() > 0.5:
            img = add_motion_blur(img)
            meta["blur"] = "motion"
        else:
            img = add_gaussian_blur(img)
            meta["blur"] = "gaussian"
        img = add_gaussian_noise(img, sigma_range=(5, 30))
        meta["noise"] = "gaussian"

    elif task == "mirnet":
        # Low light / Denoising
        img = low_light_transform(img)
        img = add_poisson_noise(img)
        meta["low_light"] = True
        meta["noise"] = "poisson"

    elif task == "ultrazoom":
        # Super-resolution downsampling
        h, w = img.shape[:2]
        scale = random.choice([2, 3, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (w, w), interpolation=cv2.INTER_NEAREST) # Up to original for training pair
        meta["sr_scale"] = scale

    elif "nima" in task:
        # General quality assessment - mix of everything
        options = [
            lambda x: add_gaussian_noise(x),
            lambda x: add_gaussian_blur(x),
            lambda x: apply_jpeg_compression(x)
        ]
        img = random.choice(options)(img)
        meta["nima_degradation"] = True

    else:
        # Generic degradation for other tasks
        img = add_gaussian_noise(img, sigma_range=(2, 15))
        img = apply_jpeg_compression(img, quality_range=(70, 95))

    return img, meta