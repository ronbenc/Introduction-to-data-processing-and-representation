from matplotlib import pyplot as plt
import numpy as np
from implementation_2 import MADSubSampler

class IRLS:
    def __init__(self, f, N, epsilon, p = 1,delta =0.0000001) -> None:
        self.f = f
        self.N = N
        self.epsilon = epsilon
        self.p = p
        self.w = np.ones(f.shape)
        self.g = np.zeros_like(f) # f^
        self.I = int(f.shape[0]/N)
        self.error = None
        self.delta = delta
    

    def workflow(self):
        while(True):
            self.g = self.get_updated_g()
            self.w = self.get_updated_w()
            prev_error = self.error
            self.error = self.calc_error()
            print('old error = {} new error = {}'.format(prev_error, self.error))
            if(prev_error is not None and np.abs((prev_error-self.error)) < self.delta):
                break

    def get_updated_g(self):
        new_g = np.zeros_like(self.g)

        for row in range(self.N):
            for col in range(self.N):
                new_g[int(row*self.I):int(((row+1)*(self.I))),int(col*self.I):int((col+1)*(self.I))] = self.get_updated_g_i(row, col)
        
        return new_g

    def get_updated_g_i(self, row, col) -> float:
        start_row = int(row*self.I)
        end_row = int((row+1)*(self.I))
        start_column = int(col*self.I)
        end_column = int((col+1)*(self.I))
        w_i = self.w[start_row:end_row,start_column:end_column]
        f_i = self.f[start_row:end_row,start_column:end_column]

        return (w_i * f_i).sum()/w_i.sum()

    def get_updated_w(self):
        normalized_w = np.abs((self.f - self.g))**(self.p-2)
        normalized_w[normalized_w > (1/self.epsilon)] = 1/self.epsilon
        return normalized_w

    def calc_error(self):
        return ((np.abs((self.f - self.g))**(self.p)).sum())/len(self.f.ravel())

if __name__ == '__main__':
    img_path = './HW1/lena_gray.gif'
    img = plt.imread(img_path)
    grayscale_img = img[:,:,0]

    signal = grayscale_img/255

    
    for k in range(1, 9):
        N = 2**k

        irls = IRLS(signal, N, 0.001, p=1)
        irls.workflow()

        mad_subsampler = MADSubSampler(grayscale_img, 2**(9-k))
        mad_subsampler.subsample()


        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(irls.g,'gray',vmin=0,vmax=1)
        axs[0].set_title("IRLS")
        axs[1].imshow(mad_subsampler.reconstructed_img,'gray',vmin=0,vmax=255)
        plt.show()
