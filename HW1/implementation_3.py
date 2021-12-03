from matplotlib import pyplot as plt
import numpy as np

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

    irls = IRLS(signal,(2**4),10000,p=4)
    irls.workflow()
    plt.imshow(irls.g,'gray',vmin=0,vmax=1)
    plt.show()