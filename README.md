# Image Denoising using Principal Component Analysis (PCA)

This Python script demonstrates the application of Principal Component Analysis (PCA) for denoising images, specifically focusing on the MNIST dataset. The code utilizes the Keras library to load and preprocess the data, followed by the introduction of noise and subsequent denoising using PCA.

## Instructions:

1. Ensure that the required libraries are installed by executing the following command:
   ```bash
   pip install keras numpy matplotlib
   ```

2. Run the script using a Python interpreter.

## Code Overview:

### 1. Loading and Visualizing Original Images
   - Load the MNIST dataset and display a subset of original images for each digit.

```python
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from matplotlib import colors as mcolors
colors = list(mcolors.cnames.keys())

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_ds = x_train.reshape((60000, 784))

imgs = []
for i in range(10):
    for j in range(y_train.shape[0]):
        if y_train[j] == i:
            imgs.append(img_ds[j, :])
            if len(imgs) % 100 == 0:
                break

a = np.array(imgs)
print(a.shape)

_20_images = []
for i in range(1000):
    if i % 100 == 1 or i % 100 == 2:
        _20_images.append(a[i, :])
_20_images = np.array(_20_images)

plt.figure(100, figsize=(40, 20))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    pixels = _20_images[i, :].reshape((28, 28))
    plt.imshow(pixels)
plt.show()
```

### 2. Denoising with PCA
   - Compute the singular value decomposition (SVD) and perform PCA on the original images.
   - Introduce random noise to a subset of pixels in the images.

```python
l = np.mean(a, axis=0)
a_zero_mean = a - l

u, s, vh = np.linalg.svd(a_zero_mean, full_matrices=True)

plt.title('Singular Values')
plt.plot(s)
plt.show()
```

### 3. Visualizing Reduced-Dimension Images
   - Display a scatter plot of images in a reduced-dimensional space (2D in this case).

```python
new_dim = 2
new_vh = vh[:new_dim, :]
aa = np.matmul(a_zero_mean, new_vh.transpose())

fig = plt.figure()
for i in range(10):
    plt.scatter(aa[i * 100:i * 100 + 2, 0], aa[i * 100:i * 100 + 2, 1], label=str(i),
                c=colors[i + 5])
plt.show()
```

### 4. Adding Noise to Images and Reapplying PCA
   - Introduce noise to a subset of pixels in the images.
   - Compute PCA on the images with added noise.

```python
noise_a = np.copy(a)
indices = sample(range(0, 783), int(0.3 * 784))
for i in range(len(indices)):
    noise = np.random.normal(0, 1, 1000)
    noise_a[:, indices[i]] = noise_a[:, indices[i]] + noise
```

### 5. Visualizing Noisy Reduced-Dimension Images
   - Display a scatter plot of images in a reduced-dimensional space (2D in this case) after adding noise.

```python
l = np.mean(noise_a, axis=0)
noise_a_zero_mean = noise_a - l

u, s, vh = np.linalg.svd(noise_a_zero_mean, full_matrices=True)

plt.title('Singular Values of Noise Images')
plt.plot(s)
plt.show()
```

### 6. Denoising Noisy Images with PCA
   - Apply PCA on the images with added noise to denoise them.

```python
new_dim = 2
new_vh = vh[:new_dim, :]
noise_aa = np.matmul(noise_a_zero_mean, new_vh.transpose())

new_dim = 10
new_vh = vh[:new_dim, :]
noise_aaa = np.matmul(np.matmul(u, np.append(np.diagflat(s), np.zeros((216, 784)), axis=0)[:, :new_dim]),
                      new_vh)
noise_aaa = noise_aaa + l

_20_images = []
for i in range(1000):
    if i % 100 == 1 or i % 100 == 2:
        _20_images.append(noise_aaa[i, :])
_20_images = np.array(_20_images)

plt.figure(100, figsize=(40, 20))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    pixels = _20_images[i, :].reshape((28, 28))
    plt.imshow(pixels)
plt.show()
```
