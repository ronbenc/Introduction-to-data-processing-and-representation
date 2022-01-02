import numpy as np
from matplotlib import pyplot as plt
from os import path

class DFT:

    def get_DFT_coeffs(img) -> np.array:
        coeffs = np.empty_like(img, dtype=complex)
        for row, array in enumerate(img):
            coeffs[row, :] = np.fft.fft(array)
        
        return coeffs

    def reconstruct_img(coeffs) -> np.array:
        img = np.empty_like(coeffs)
        for row, array in enumerate(coeffs):
            img[row, :] = np.fft.ifft(array)

        return img.real


if __name__ == '__main__':
    # load images
    mandril_org_path = path.join('HW3', 'images_and_audios', 'images and audios', 'mandril_original.png')
    mandril_org_img = plt.imread(mandril_org_path)

    mandril_dist_path = path.join('HW3', 'images_and_audios', 'images and audios', 'mandril_distorted.png')
    mandril_dist_img = plt.imread(mandril_dist_path)

    butterfly_org_path = path.join('HW3', 'images_and_audios', 'images and audios', 'Butterfly_.png')
    butterfly_org_img = plt.imread(butterfly_org_path)

    # get the DFT basus coefficients for the original and distorted image
    A = DFT.get_DFT_coeffs(mandril_org_img) # alphas
    B = DFT.get_DFT_coeffs(mandril_dist_img) # betas

    # Assert that A is full rank
    rank_A = np.linalg.matrix_rank(A)
    assert (rank_A, rank_A) == mandril_org_img.shape, "A is not full rank"

    # find C using least squares
    C = np.linalg.lstsq(A, B, rcond=None)[0]

    # Distort the original image rows using C
    B_aprrox = np.matmul(A, C)

    # Compare the results
    mandril_dist_aprrox_img = DFT.reconstruct_img(B_aprrox)

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("given distorted image")
    axs[0].imshow(mandril_dist_img, 'gray', vmin = 0, vmax = 1)
    axs[1].set_title("aprroximated distorted image")
    axs[1].imshow(mandril_dist_aprrox_img, 'gray', vmin = 0, vmax = 1)
    MSE = np.square(np.subtract(mandril_dist_img,mandril_dist_aprrox_img)).mean()
    plt.text(-100, -50, "MSE = {}".format(MSE))
    plt.show()

    # Distort and display 'butterfly' image
    D = DFT.get_DFT_coeffs(butterfly_org_img)
    E_approx = np.matmul(D, C)
    bitterfly_dist_aprrox_img = DFT.reconstruct_img(E_approx)
    plt.title("aprroximated distorted image")
    plt.imshow(bitterfly_dist_aprrox_img, 'gray', vmin = 0, vmax = 1)
    plt.show()
    

    

    
