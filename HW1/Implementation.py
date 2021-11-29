from matplotlib import pyplot as plt
import numpy as np


def my_round(x):
    epsilon = 0.0000001
    return np.round(x+epsilon)
class MaxLloyd():
    
    def __init__(self, epsilon,hist,decisions):
        self.hist = hist
        self.pdf = hist[0] / hist[0].sum()
        self.epsilon = epsilon
        self.d = np.array(decisions).astype(int)
        self.error = None
        self.r = None
        self.k = len(decisions) - 1
        
    def calc_representation(self) -> list:
        reps = []
        for i in range(self.k):
            elements_in_range = self.hist[0][self.d[i]:self.d[i+1]-1].sum()
            if elements_in_range == 0:
                reps.append(int(my_round((self.d[i]+self.d[i+1])/2)))
            else:
                # x_in_range = self.hist[0][self.d[i]:self.d[i+1]-1]
                # hist_in_range = self.hist[1][self.d[i]:self.d[i+1]-1]
                # reps.append((x_in_range*hist_in_range).sum()/elements_in_range)
                start = self.d[i]
                end = max(self.d[i],self.d[i+1]-1)
                reps.append(int(my_round((self.hist[0][start:end]*self.hist[1][start:end]).sum()/elements_in_range)))
        return reps
        # return [int(my_round((self.hist[0][self.d[i]:self.d[i+1]]*self.hist[1][self.d[i]:self.d[i+1]]).sum()/max(1,self.hist[0][self.d[i]:self.d[i+1]].sum()))) for i in range(self.k)]

    def calc_decisions(self) -> list:
        return [0] + [int(my_round((self.r[i] + self.r[i+1])/2)) for i in range(self.k-1)] + [int(self.hist[1][-1])]
        

    def calc_error(self):
        error_sum = 0
        for i in range(self.k):
            for x in range(self.d[i], self.d[i+1]):
                error_sum += (((x-self.r[i])**2)*self.pdf[x])
        
        return error_sum

    def WorkFlow(self):
        while True:
            self.r = self.calc_representation()
            self.d = self.calc_decisions()
            old_error = self.error
            self.error = self.calc_error()
            if old_error is not None and old_error - self.error < self.epsilon:
                break
        return (list(self.r), list(self.d))


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

def get_decisions_and_representations(num_bits,low,high):
    k = 2**num_bits
    delta = (high-low)/k
    decisions = [int(low+i*delta) for i in range(0,k+1)]
    representations = [int(low + (i-(1/2))*delta) for i in range(1,k+1)]
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

def plot_max_lloyd(b_list,epsilon,hist):
    decisions = []
    representations = []
    error = []
    for bit in b_list:
        init_decision,_ = get_decisions_and_representations(bit,0,256)
        max_lloyd = MaxLloyd(epsilon,hist,init_decision)
        r,d = max_lloyd.WorkFlow()
        decisions.append(list(d))
        representations.append(list(r))
        error.append(max_lloyd.error)
    pass


if __name__ == '__main__':
    img_path = './HW1/lena_gray.gif'
    img = plt.imread(img_path)
    grayscale_img = img[:,:,0]
    hist = plt.hist(grayscale_img.ravel(),bins=256,range=[0,256])
    plt.show()
    max_num_bits = 8

    b_list = [bits for bits in range(1,max_num_bits+1)]

    # #2.1
    # plot_MSE(b_list)

    # #2.b
    # plot_reps_and_dec(b_list)

    plot_max_lloyd(b_list,0.001,hist)



   




    