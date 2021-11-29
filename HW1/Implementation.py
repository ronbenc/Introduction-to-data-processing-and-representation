from matplotlib import pyplot as plt
import numpy as np

class MaxLloyd():
    
    def __init__(epsilon,hist,decisions):
        self.pdf = hist
        self.epsilon = epsilon
        self.decisions = decisions
        self.error = None
        self.representations = None

    def calc_representation(self) -> list:
        return

    def calc_decisions(self) -> list:
        return

    def calc_errors(self):
        return

    def WorkFlow(self):
        while self.error > self.epsilon:
            self.representations = self.calc_representation()
            self.decisions = self.calc_decisions()
            self.error = self.calc_error()
        return (self.representations, self.decisions)


def uniform_quantize_function(low, delta, x):
    return low+((np.floor(((x-low)/delta))+(1/2))*delta)

def uniform_quantize(img,num_bits,low,high):
    k = 2**num_bits
    delta = (high-low)/k
    return uniform_quantize_function(low,delta,img)
    # return (get_decisions_and_representations(delta,k,low),uniform_quantize_function(low,delta,img))

def compute_MSE(orig_img, q_img):
    N = orig_img.ravel().shape[0]
    return (np.power((orig_img-q_img).astype(float), 2)).sum()/N

def get_decisions_and_representations(delta,k,low):
    decisions = [low+i*delta for i in range(0,k+1)]
    representations = [low + (i-(1/2))*delta for i in range(1,k+1)]
    return (decisions,representations)

def plot_reps_and_dec(b_list):
    x = [i for i in range(0,256)]
    for bit in b_list:
        plt.plot(x,uniform_quantize(np.array(x),bit,0,256))
        plt.show()

def plot_MSE(b_list):
    q_list = []

    for bit in b_list:
        quantized_img = uniform_quantize(grayscale_img,bit,0,256)
        q_list.append(compute_MSE(grayscale_img, quantized_img))
        
        # decisions.append(dec_and_reps[0])
        # representations.append(dec_and_reps[1])
        # plt.imshow(quantized_img,'gray')
        # plt.show()
    plt.plot(b_list, q_list)
    plt.show()

if __name__ == '__main__':
    img_path = './HW1/lena_gray.gif'
    img = plt.imread(img_path)
    grayscale_img = img[:,:,0]
    plt.hist(grayscale_img.ravel(),bins=256,range=[0,256])
    plt.show()
    max_num_bits = 8

    b_list = [bits for bits in range(1,max_num_bits+1)]

    #2.1
    plot_MSE(b_list)

    #2.b
    plot_reps_and_dec(b_list)

    
   




    