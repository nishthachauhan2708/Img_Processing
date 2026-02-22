import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image
img = cv2.imread('image.tif', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found. Check the path.")

img = np.float32(img)

# -----------------------------------
# 2. Add Gaussian Noise
# -----------------------------------
mean = 0
sigma = 25
gaussian = np.random.normal(mean, sigma, img.shape)
noisy_gaussian = img + gaussian

noisy_gaussian = np.clip(noisy_gaussian, 0, 255)
noisy_gaussian = np.uint8(noisy_gaussian)

# -----------------------------------
# 3. Add Salt & Pepper Noise
# -----------------------------------
noisy_sp = img.copy()
prob = 0.02  # noise probability

# Salt noise
salt = np.random.rand(*img.shape) < prob
# Pepper noise
pepper = np.random.rand(*img.shape) < prob

noisy_sp[salt] = 255
noisy_sp[pepper] = 0
noisy_sp = np.uint8(noisy_sp)

# -----------------------------------
# 4. Apply Mean Filter (3x3)
# -----------------------------------
mean_gaussian = cv2.blur(noisy_gaussian, (3, 3))
mean_sp = cv2.blur(noisy_sp, (3, 3))

# -----------------------------------
# 5. Apply Median Filter (3x3)
# -----------------------------------
median_gaussian = cv2.medianBlur(noisy_gaussian, 3)
median_sp = cv2.medianBlur(noisy_sp, 3)

# -----------------------------------
# 6. Display Results
# -----------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(3,3,1), plt.imshow(img, cmap='gray')
plt.title("Original"), plt.axis('off')

plt.subplot(3,3,2), plt.imshow(noisy_gaussian, cmap='gray')
plt.title("Gaussian Noise"), plt.axis('off')

plt.subplot(3,3,3), plt.imshow(mean_gaussian, cmap='gray')
plt.title("Mean Filter (Gaussian)"), plt.axis('off')

plt.subplot(3,3,4), plt.imshow(median_gaussian, cmap='gray')
plt.title("Median Filter (Gaussian)"), plt.axis('off')

plt.subplot(3,3,5), plt.imshow(noisy_sp, cmap='gray')
plt.title("Salt & Pepper Noise"), plt.axis('off')

plt.subplot(3,3,6), plt.imshow(mean_sp, cmap='gray')
plt.title("Mean Filter (S&P)"), plt.axis('off')

plt.subplot(3,3,7), plt.imshow(median_sp, cmap='gray')
plt.title("Median Filter (S&P)"), plt.axis('off')

plt.tight_layout()
plt.show()
