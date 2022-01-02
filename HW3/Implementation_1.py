import numpy as np
from matplotlib import pyplot as plt
from os import path

def get_DFT_coeffs(img) -> np.array:
    coeffs = np.empty_like(img, dtype=complex)
    for row, array in enumerate(img):
        coeffs[row, :] = np.fft.fft(array)
    
    return coeffs

# for testing
def reconstruct_img(coeffs) -> np.array:
    img = np.empty_like(coeffs)
    for row, array in enumerate(coeffs):
        img[row, :] = np.fft.ifft(array)

    return img.real

if __name__ == '__main__':
    mandril_org_path = path.join('HW3', 'images_and_audios', 'images and audios', 'mandril_original.png')
    mandril_org = plt.imread(mandril_org_path)

    mandril_dist_path = path.join('HW3', 'images_and_audios', 'images and audios', 'mandril_distorted.png')
    mandril_dist = plt.imread(mandril_dist_path)

    alphas = get_DFT_coeffs(mandril_org)
    betas = get_DFT_coeffs(mandril_dist)
