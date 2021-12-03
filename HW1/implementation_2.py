from abc import abstractmethod
from matplotlib import pyplot as plt
import numpy as np

class SubSampler:
    def __init__(self, img, d) -> None:
        self.img = img
        self.d = d
        self.img_shape = img.shape
        self.subsampled_shape = tuple([int(i/d) for i in self.img_shape])
        self.subsampled_img = np.zeros(self.subsampled_shape)
        self.error = 0

    

    @abstractmethod
    def compute(self):
        ...
    
    def subsample(self):
        self.error = 0

        for row in range(self.subsampled_shape[0]):
            for column in range(self.subsampled_shape[1]):
                error , value = self.compute(self.img[self.d*row:self.d*(row+1),self.d*column:self.d*(column+1)])
                self.subsampled_img[row,column] = value
                self.error = self.error + error

        self.error = self.error/len(self.img.ravel())
        return self.subsampled_img
    

    def reconstruct(self):
        ...


class MSESubSampler(SubSampler):
    def __init__(self, img, d) -> None:
        super().__init__(img, d)

    def compute(self,sub_img):
        mean = int(sub_img.mean())
        error_sum = (sub_img-mean).sum()
        return error_sum,mean

class MADSubSampler(SubSampler):
    def __init__(self, img, d) -> None:
        super().__init__(img, d)

    def compute(self,sub_img):
        median = int(np.median(sub_img))
        error_sum = (sub_img-median).sum()
        return error_sum,median

def plot_MSE():
    ...
    
def plot_MAD():
    ...

if __name__ == '__main__':
    img_path = './HW1/lena_gray.gif'
    img = plt.imread(img_path)
    grayscale_img = img[:,:,0]

    mse = MSESubSampler(img, 2)