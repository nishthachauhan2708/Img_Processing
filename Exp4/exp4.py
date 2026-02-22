import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Load image in grayscale
image = cv2.imread('image.tif', 0)

if image is None:
    raise ValueError("Image not found. Check the file path.")

# 2️⃣ Convert image to frequency domain using FFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Magnitude Spectrum (for visualization)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# 3️⃣ Create Low Pass and High Pass Masks
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Create ideal circular low-pass mask
mask_low = np.zeros((rows, cols), np.uint8)
radius = 50   # You can change this value

for i in range(rows):
    for j in range(cols):
        if (i - crow) ** 2 + (j - ccol) ** 2 <= radius ** 2:
            mask_low[i, j] = 1

# High-pass mask is inverse of low-pass
mask_high = 1 - mask_low

# 4️⃣ Apply Low Pass Filter
fshift_low = fshift * mask_low
f_ishift_low = np.fft.ifftshift(fshift_low)
img_low = np.fft.ifft2(f_ishift_low)
img_low = np.abs(img_low)

# 5️⃣ Apply High Pass Filter
fshift_high = fshift * mask_high
f_ishift_high = np.fft.ifftshift(fshift_high)
img_high = np.fft.ifft2(f_ishift_high)
img_high = np.abs(img_high)

# 6️⃣ Display Results
plt.figure(figsize=(12,8))

plt.subplot(231)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(232)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis('off')

plt.subplot(233)
plt.imshow(mask_low, cmap='gray')
plt.title("Low Pass Mask")
plt.axis('off')

plt.subplot(234)
plt.imshow(img_low, cmap='gray')
plt.title("Low Pass Filtered Image")
plt.axis('off')

plt.subplot(235)
plt.imshow(img_high, cmap='gray')
plt.title("High Pass Filtered Image")
plt.axis('off')

plt.tight_layout()
plt.show()
