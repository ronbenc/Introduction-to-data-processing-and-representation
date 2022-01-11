import numpy as np
from matplotlib import pyplot as plt
from os import path
from numpy.core.function_base import linspace
from scipy.io import wavfile
from scipy.linalg import dft

DISTORTION_PERIOD = 512

if __name__ == '__main__':
    # load skycastle waves
    skycastle_org_path = path.join('images_and_audios', 'images and audios', 'skycastle.wav')
    skycastle_org_data = wavfile.read(skycastle_org_path)[1]

    skycastle_dist_path = path.join('images_and_audios', 'images and audios', 'skycastle-distortion.wav')
    skycastle_dist_data = wavfile.read(skycastle_dist_path)[1]

    # reshape to matrix form
    skycastle_org_data_len = skycastle_org_data.shape[0]
    skycastle_dist_data_len = skycastle_dist_data.shape[0]
    assert skycastle_org_data_len == skycastle_dist_data_len

    skycastle_org_mat = np.reshape(skycastle_org_data, (skycastle_org_data_len//DISTORTION_PERIOD, DISTORTION_PERIOD))
    skycastle_dist_mat = np.reshape(skycastle_dist_data,  (skycastle_dist_data_len//DISTORTION_PERIOD, DISTORTION_PERIOD))

    # get the DFT basis coefficients for the original and distorted image
    DFT_mat = dft(n=DISTORTION_PERIOD, scale="sqrtn")
    IDFT_mat = DFT_mat.conj().T

    A = np.matmul(skycastle_org_mat, DFT_mat)
    B = np.matmul(skycastle_dist_mat, DFT_mat)

    # find H using least squares, This functional map will remove the destortion when applied on a distroted wave
    H = np.linalg.lstsq(B, A, rcond=None)[0]

    # load totoro waves
    totoro_org_path = path.join('images_and_audios', 'images and audios', 'totoro.wav')
    totoro_org_data = wavfile.read(totoro_org_path)[1]

    totoro_dist_path = path.join('images_and_audios', 'images and audios', 'totoro-distortion.wav')
    totoro_dist_data = wavfile.read(totoro_dist_path)[1]

    # reshape to matrix form
    totoro_org_data_len = totoro_org_data.shape[0]
    totoro_dist_data_len = totoro_dist_data.shape[0]
    assert totoro_org_data_len == totoro_dist_data_len 

    totoro_org_matrix = np.reshape(totoro_org_data, (totoro_org_data_len//DISTORTION_PERIOD, DISTORTION_PERIOD))
    totoro_dist_matrix = np.reshape(totoro_dist_data, (totoro_dist_data_len//DISTORTION_PERIOD, DISTORTION_PERIOD))

    # Remove distortion from totoro and compare to original
    D = np.matmul(totoro_dist_matrix, DFT_mat)
    C_approx = np.matmul(D, H)
    totoro_approx_data = np.ravel((np.matmul(C_approx, IDFT_mat).real))
    MSE = np.square(np.subtract(totoro_org_data, totoro_approx_data)).mean()
    wavfile.write("Aprroximated_totoro.wav", 48000, totoro_approx_data.astype(np.int16))
    
    plt.title("MSE = {}".format(MSE))
    plt.plot(totoro_org_data, 'r')
    plt.plot(totoro_approx_data, 'b')
    plt.legend(["original", "Approximated"])
    plt.show()

