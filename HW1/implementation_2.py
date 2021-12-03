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
        self.name = "MSE"

    def compute(self,sub_img):
        mean = sub_img.mean()
        error_sum = ((sub_img-mean)**2).sum()
        return error_sum, int(mean)

class MADSubSampler(SubSampler):
    def __init__(self, img, d) -> None:
        super().__init__(img, d)
        self.name = "MAD"

    def compute(self,sub_img):
        median = np.median(sub_img)
        error_sum = (np.abs(sub_img-median)).sum()
        return error_sum, int(median)

def plot_sampling(img, Sampler):
    d_list = [2**i for i in range(1,9)]
    error_list = []
    fig,axs = plt.subplots(4, 2)
    for i, d in enumerate(d_list):
        subsampler = Sampler(img, d)
        sub_sampled_img = subsampler.subsample()
        error_list.append(subsampler.error)
        axs[i%4,int(i/4)].imshow(sub_sampled_img,'gray', vmin = 0, vmax = 255)
        axs[i%4,int(i/4)].set_title('sub-sampling factor = {}'.format(d))

    plt.show()
    plt.plot(d_list, error_list)
    plt.title(subsampler.name)
    plt.xlabel("sub-sampling factor D")
    plt.ylabel("error")
    plt.show()


if __name__ == '__main__':
    img_path = './HW1/lena_gray.gif'
    img = plt.imread(img_path)
    grayscale_img = img[:,:,0]

    plot_sampling(grayscale_img, MSESubSampler)
    plot_sampling(grayscale_img, MADSubSampler)