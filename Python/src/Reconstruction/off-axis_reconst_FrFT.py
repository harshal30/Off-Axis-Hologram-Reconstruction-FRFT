import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from numpy.fft import fftshift
from tqdm import tqdm # for progress bars
from statistics import mean, stdev
import torch
import frft
# import frft_gpu as frft_g
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import time
from numpy.linalg import svd
from numpy.polynomial.polynomial import Polynomial
from skimage.restoration import unwrap_phase
# from kamui import unwrap_dimensional
from scipy.fftpack import dct, idct
from utils import *

Ho = cv2.imread('./data/raw_hologram.tif',0)
Ho_upd = np.pad(Ho, pad_width=((250, 250), (250, 250)), mode='constant', constant_values=0)
M, N = Ho_upd.shape
# Visualization
plt.figure()
plt.imshow(Ho_upd, cmap='gray')
plt.title('Hologram')
plt.show()

Ho_shift = fftshift( Ho_upd )

fobj_1d = frft.frft( Ho_shift, 0.88 )
fobj_1d = fftshift( fobj_1d )  
plt.imshow(np.log(np.absolute( fobj_1d )/np.absolute( fobj_1d.max() )))

spectrum_abs = np.absolute(fobj_1d)
maximum = np.max(spectrum_abs[:])
y0,x0 = np.where(spectrum_abs==maximum)

shift_x = -(x0 - (N / 2))
shift_y = (N / 2) - y0

# Apply the circular shift
spectrum2 = np.roll(np.roll(fobj_1d, int(shift_y), axis=0), int(shift_x), axis=1)
plt.imshow(np.log(np.absolute( spectrum2 )))

image_size = (M, N)
center_x, center_y = image_size[0]/2, image_size[1]/2
radius = 150
y, x = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
mask = distance <= radius
plt.imshow(mask)
spectrum3_upd1 = spectrum2 * mask
plt.imshow(np.log(np.absolute( spectrum3_upd1 )))

Psi_pca = PCA_aberration_comp(spectrum3_upd1)



# Hologram reconstruction parameters

z_start = -50e-6
z_end = 20e-6
z_step = 5e-6
p = 5.5e-6
p = p/50
lambda_ = 532e-9  # Wavelength in meters
h1 = M * p
h2 = N * p

S = round((z_end - z_start) / z_step)
reconstructionA = np.zeros((M, N, S), dtype=np.complex128)
localAMSA = np.zeros(S)
localGRAA = np.zeros(S)
localDFSA = np.zeros(S)


for ii in range(S):
    z0 = z_start + ii * z_step

    prop = Propagator(M, N, lambda_, h1, h2, -z0)
    recA = fftshift(ifft2(Psi_pca * prop))
    # recA = fftshift(ifft2(spectrum3_upd1 * prop))
    reconstructionA[:, :, ii] = recA
    localAMSA[ii] = funcAutoFocusAMS(np.abs(recA))
    localGRAA[ii] = funcAutoFocusGRA(np.abs(recA))
    localDFSA[ii] = funcAutoFocusDFS(np.abs(recA))
    # localSVDA[ii] = funcAutoFocusSVD(np.abs(recA), 103)
    plt.imshow(np.abs(recA), cmap='gray')
    plt.title(f'The reconstruction distance is {z0}')
    plt.pause(0.05)

# Normalize local metrics
localAMSA = (localAMSA - np.min(localAMSA)) / (np.max(localAMSA) - np.min(localAMSA))
localGRAA = (localGRAA - np.min(localGRAA)) / (np.max(localGRAA) - np.min(localGRAA))
localDFSA = (localDFSA - np.min(localDFSA)) / (np.max(localDFSA) - np.min(localDFSA))
# localSVDA = (localSVDA - np.min(localSVDA)) / (np.max(localSVDA) - np.min(localSVDA))

# Create v2
v1 = np.arange(-50, 20, 5)


# Plot local metrics
plt.figure()
plt.plot(v1, localAMSA, 'r-*', linewidth=1.5, markersize=4, linestyle=':', label='AMS')
plt.plot(v1, localGRAA, 'g-D', linewidth=1.5, markersize=4, linestyle=':', label='GRA')
plt.plot(v1, localDFSA, 'b-h', linewidth=1.5, markersize=4, linestyle=':', label='DFS')
# plt.plot(v2, localSVDA, 'm-h', linewidth=1.5, markersize=4, linestyle=':', label='SVD')
plt.xlabel('z(um)', fontsize=12, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.axis('tight')

# Find the best reconstruction
R = np.argmax(localAMSA)
# R = np.argmin(localGRAA)
# R = np.argmax(localDFSA)
# R = 10
phase = np.angle(reconstructionA[:, :, R])

plt.figure()
plt.imshow(phase, cmap='gray')
plt.title(f'The reconstruction distance is {v1[R]} um')

plt.figure()
plt.imshow(np.absolute(reconstructionA[:, :, R]), cmap='gray')
plt.title(f'The reconstruction distance is {v1[R]} um')

wrapped_phase = phase[1001:2000,501:1700]
plt.figure()
plt.imshow(wrapped_phase, cmap='gray')
plt.colorbar()
plt.axis('on')
plt.show()

image_unwrapped2 = unwrap_TIE(wrapped_phase)
plt.figure()
plt.imshow(image_unwrapped2, cmap='gray')
plt.colorbar()
plt.axis('on')
plt.show()



# image_unwrapped2 = unwrap_dimensional(wrapped_phase)
# plt.figure()
# plt.imshow(image_unwrapped2, cmap='gray')

temp = image_unwrapped2
M, N = temp.shape

mask = np.ones_like(temp)
F = zernike_func_cartesian_coordinate(temp)
a1, Z1 = zernike_coeffs_cartesian(temp, F, mask)

Z2 = Z1 @ a1
Z2_sim = Z2.reshape(M, N)
True_Phase = temp - Z2_sim

# Visualize the true phase
plt.figure()
plt.imshow(True_Phase, cmap='gray')
plt.colorbar()
plt.title('True Phase')
plt.axis('on')
plt.show()