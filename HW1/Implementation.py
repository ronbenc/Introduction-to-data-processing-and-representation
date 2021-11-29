from matplotlib import pyplot as plt
import numpy as np

def uniform_quantize_function(low,delta,x):
    return low+((np.floor(((x-low)/delta))+(1/2))*delta)

def uniform_quantize(img,num_bits,low,high):
    k = 2**num_bits
    delta = (high-low)/k
    return uniform_quantize_function(low,delta,img)




if __name__ == '__main__':
    img_path = './HW1/lena_gray.gif'
    img = plt.imread(img_path)
    grayscale_img = img[:,:,0]
    plt.hist(grayscale_img.ravel(),bins=256,range=[0,256])
    max_num_bits = 8
    
    for bits in range(1,max_num_bits+1):
        quantized_img = uniform_quantize(grayscale_img,bits,0,256)
        plt.imshow(quantized_img,'gray')
        plt.title('number of bits = {}'.format(bits))
        plt.show()