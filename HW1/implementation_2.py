from abc import abstractmethod
from matplotlib import pyplot as plt
import numpy as np

class SubSampler:
    def __init__(self, img, d) -> None:
        self.img = img
        self.d = d
        self.subsampled_img = None

    @abstractmethod
    def compute(self):
        ...
    
    def subsample(self):
        ...
    

    def reconstruct(self):
        ...


class MSESubSampler(SubSampler):
    def __init__(self, img, d) -> None:
        super().__init__(img, d)

    def compute(self):



if __name__ == '__main__':
    img_path = './HW1/lena_gray.gif'
    img = plt.imread(img_path)
    grayscale_img = img[:,:,0]

    mse = MSESubSampler(img, 2)