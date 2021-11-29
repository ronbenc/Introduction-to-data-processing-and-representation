from matplotlib import pyplot as plt
import numpy as np

def uniform_quantize_function(low, delta, x):
    return low+((np.floor(((x-low)/delta))+(1/2))*delta)

def uniform_quantize(img,num_bits,low,high):
    k = 2**num_bits
    delta = (high-low)/k
    return uniform_quantize_function(low,delta,img)

def compute_MSE(orig_img, q_img):
    N = orig_img.ravel().shape[0]
    return (np.power((orig_img-q_img).astype(float), 2)).sum()/N


if __name__ == '__main__':
    img_path = './HW1/lena_gray.gif'
    img = plt.imread(img_path)
    grayscale_img = img[:,:,0]
    plt.hist(grayscale_img.ravel(),bins=256,range=[0,256])
    plt.show()
    max_num_bits = 8
    
    b_list = [bits for bits in range(1,max_num_bits+1)]
    q_list = []


    for bit in b_list:
        quantized_img = uniform_quantize(grayscale_img,bit,0,256)
        q_list.append(compute_MSE(grayscale_img, quantized_img))

    plt.plot(b_list, q_list)
    plt.show()
        