import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
import cv2

class Phi:
    """A class to represent the given function phi"""

    def __init__(self, A: float, w_x: float, w_y: float) -> None:
        self.A = A
        self.w_x = w_x
        self.w_y = w_y

        # values caculated analiticly 
        self.val_high = self.A
        self.val_low = -self.A
        self.val_range = self.val_high - self.val_low
        self.horizontal_derivative_energy = (math.pi * self.A * self.w_x)**2
        self.vertical_derivative_energy = (math.pi * self.A * self.w_y)**2
    
    def compute(self, x: float, y: float) -> float:
        return self.A*math.cos(2*math.pi*self.w_x*x)*math.sin(2*math.pi*self.w_y*y)
    
    def __repr__(self) -> str:
        return "{}cos(2pi*{}*x)sin(2pi*{}*y)".format(self.A, self.w_x, self.w_y)


class ApproximatePhi:
    """A class to represent the an approximation of the function phi"""

    def __init__(self, phi: Phi, hor_samples: int, ver_samples: int) -> None:
        self.phi = phi
        self.hor_samples = hor_samples
        self.ver_samples = ver_samples

        # values caculated numericaly  
        self._approximate_phi()
        self.val_high = 0
        self.val_low = 0

        self.approximate_val_range()
        self.approximate_vertical_derivative_energy()
        self.approximate_horizontal_derivative_energy()
        

    def _approximate_phi(self) -> np.array:
        self.approximated_phi = np.empty((self.hor_samples, self.ver_samples))
        for x in range(self.hor_samples):
            for y in range(self.ver_samples):
                self.approximated_phi[x, y] = self.phi.compute((x/self.hor_samples), (y/self.ver_samples))
    
    def _approximate_val_high(self):
        self.val_high = np.amax(self.approximated_phi)

    def _approximate_val_low(self):
        self.val_low = np.amin(self.approximated_phi)

    def approximate_val_range(self):
        self._approximate_val_high()
        self._approximate_val_low()
        self.val_range = self.val_high - self.val_low

    def _approximate_phi_vertical_derivative(self):
        self.ver_derivative = np.zeros((self.hor_samples, self.ver_samples))
        for x in range(self.hor_samples):
            for y in range(self.ver_samples):
                self.ver_derivative[x, y] = \
                    ((self.approximated_phi[x, (y + 1) % self.ver_samples] - self.approximated_phi[x, y])*self.ver_samples) # phi is cyclic for T=1

    def _approximate_phi_horizontal_derivative(self):
        self.hor_derivative = np.zeros((self.hor_samples, self.ver_samples))
        for y in range(self.ver_samples):
            for x in range(self.hor_samples):
                self.hor_derivative[x, y] = \
                    ((self.approximated_phi[(x + 1) % self.hor_samples, y] - self.approximated_phi[x, y])*self.hor_samples) # phi is cyclic for T=1

    def approximate_vertical_derivative_energy(self):
        self._approximate_phi_vertical_derivative()
        self.ver_derivative_energy = np.sum(self.ver_derivative**2)/(self.hor_samples * self.ver_samples)

    def approximate_horizontal_derivative_energy(self):
        self._approximate_phi_horizontal_derivative()
        self.hor_derivative_energy = np.sum(self.hor_derivative**2)/(self.hor_samples * self.ver_samples)
class BitAllocator:
    def __init__(self, B: int, aprox_phi: ApproximatePhi) -> None:
        self.B = B
        self.aprox_phi = aprox_phi
    
    def calc_MSE(self, n_x, n_y, b):
        x_error = (1/12)*self.aprox_phi.hor_derivative_energy/(n_x**2)
        y_error = (1/12)*self.aprox_phi.ver_derivative_energy/(n_y**2)
        b_error = (1/12)*(self.aprox_phi.val_range**2)/(4**b)
        return x_error + y_error + b_error

    def search_params(self) -> tuple((float, float, float)):
        min_error = float('inf')
        best_params = None
        for n_x in np.linspace(1, int(self.B), int(self.B*10)):
            for n_y in np.linspace(1, int(self.B/n_x), int((self.B/n_x)*10)):
                b = min(self.B/(n_x*n_y), 100)
                curr_error = self.calc_MSE(n_x, n_y, b)
                if curr_error < min_error:
                    min_error = curr_error
                    best_params = (n_x, n_y, b)

        return best_params

    def calc_params(self) -> tuple((float, float, float)):
        sqrt_energy = np.sqrt(self.aprox_phi.hor_derivative_energy*self.aprox_phi.ver_derivative_energy)
        b = (1/2)*np.log2(self.B*np.log(4)*((self.aprox_phi.val_range**2)/(2*sqrt_energy)))

        sqrt_c = np.sqrt(self.B/b)
        n_x = np.power((self.aprox_phi.hor_derivative_energy/self.aprox_phi.ver_derivative_energy), (1/4))*sqrt_c
        n_y = np.power((self.aprox_phi.ver_derivative_energy/self.aprox_phi.hor_derivative_energy), (1/4))*sqrt_c

        return (n_x, n_y, b)

    def round_to_opt_params(self, b) -> tuple((int, int, int)):
        b_f = int(np.floor(b))
        b_c = int(np.ceil(b))
        best_params = None
        min_error = float('inf')

        for b in [b_f, b_c]:
            for n_x in range(1, int(self.B/b)+1):
                n_y = int((self.B/b)/n_x)
                curr_error = self.calc_MSE(n_x, n_y, b)
                if curr_error < min_error:
                    min_error = curr_error
                    best_params = (n_x, n_y, b)

        return best_params

class Reconstarctor:
    def __init__(self, img, n_x, n_y, b) -> None:
        self.img = img
        self.n_x = n_x
        self.n_y = n_y
        self.b = b

        self.img_shape = img.shape
        self.val_low = np.min(img)
        self.val_high = np.max(img)

        self.q_samples = 2**b

        self.delta = (self.val_high - self.val_low)/self.q_samples

        self.recontructed_img = np.zeros(img.shape)
    
    def recontruct(self):
        y_step = int(self.img_shape[0]/self.n_y)
        x_step = int(self.img_shape[1]/self.n_x)
        reduced_img = block_reduce(self.img, block_size=(y_step, x_step), func=np.mean)
        resized_img = cv2.resize(reduced_img, self.img.shape, interpolation=cv2.INTER_NEAREST)
        self.recontructed_img = self.uniform_quantize_function(self.val_low, self.delta, resized_img)

        # for row in range(self.n_y):
        #     for col in range(self.n_x):
        #         # print(row*y_step, (row+1)*y_step)
        #         # print(col*x_step, (col+1)*x_step)
        #         value = self.img[row*y_step: (row+1)*y_step, col*x_step: (col+1)*x_step].mean()
                    
        #         # self.subsampled_img = value
        #         self.recontructed_img[row*y_step: (row+1)*y_step, col*x_step: (col+1)*x_step] = self.uniform_quantize_function(self.val_low, self.delta, value)

    @staticmethod
    def uniform_quantize_function(low, delta, x):
        return low+((np.floor(((x-low)/delta))+(1/2))*delta)



def run_sections(A: float, w_x: float, w_y: float, hor_samples: int, ver_samples: int):
    phi = Phi(A, w_x, w_y)
    aprox_phi = ApproximatePhi(phi, hor_samples, ver_samples)
    aprox_img = aprox_phi.approximated_phi
    plt.imshow(aprox_img, 'gray', vmin = -phi.A, vmax = phi.A)
    plt.title("Approximated phi for {}*{} samples".format(aprox_phi.hor_samples, aprox_phi.ver_samples))
    plt.show()

    print("approximated vertical derivative energy: {}".format(aprox_phi.ver_derivative_energy))
    print("approximated horizontal derivative energy: {}".format(aprox_phi.hor_derivative_energy))
    print("approximated value range: {}".format(aprox_phi.val_range))

    for B in [5000, 50000]:
        bit_allocator = BitAllocator(B, aprox_phi)
        calculated_params = bit_allocator.calc_params()
        opt_params = bit_allocator.round_to_opt_params(calculated_params[2])
        print("Nx, Ny and b obtained by optimizing the bit-allocation for budget = {} are: {}, {} and {}".format(B, calculated_params[0], calculated_params[1], calculated_params[2]))
        reconstructor = Reconstarctor(aprox_img, opt_params[0], opt_params[1], opt_params[2])
        reconstructor.recontruct()
        plt.imshow(reconstructor.recontructed_img, 'gray')
        plt.title("Reconstructed image for {}".format(opt_params))
        plt.show()

    for B in [5000, 50000]:
        bit_allocator = BitAllocator(B, aprox_phi)
        evaluated_params = bit_allocator.search_params()
        opt_params = bit_allocator.round_to_opt_params(evaluated_params[2])
        print("Nx, Ny and b obtained s by practically evaluating the bit-allocation MSE for many combinations of parameters for budget = {} are: {}, {} and {}".format(B, evaluated_params[0], evaluated_params[1], evaluated_params[2]))
        reconstructor = Reconstarctor(aprox_img, opt_params[0], opt_params[1], opt_params[2])
        reconstructor.recontruct()
        plt.imshow(reconstructor.recontructed_img, 'gray')
        plt.title("Reconstructed image for {}".format(opt_params))
        plt.show()


if __name__ == '__main__':
    run_sections(2500, 2, 7, 512, 512)
    run_sections(2500, 7, 2, 512, 512)