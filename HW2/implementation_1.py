import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import lambertw

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
        self.n_x = hor_samples
        self.n_y = ver_samples

        # values caculated numericaly  
        self._approximate_phi()
        self.val_high = 0
        self.val_low = 0
        

    def _approximate_phi(self) -> np.array:
        self.approximated_phi = np.zeros((self.n_x, self.n_y))
        for x in range(self.n_x):
            for y in range(self.n_y):
                self.approximated_phi[x, y] = self.phi.compute((x/self.n_x), (y/self.n_y))
    
    def _approximate_val_high(self):
        self.val_high = np.amax(self.approximated_phi)

    def _approximate_val_low(self):
        self.val_low = np.amin(self.approximated_phi)

    def approximate_val_range(self):
        self._approximate_val_high()
        self._approximate_val_low()
        self.val_range = self.val_high - self.val_low

    def _approximate_phi_vertical_derivative(self):
        self.approximated_phi_vertical_derivative = np.zeros((self.n_x, self.n_y))
        for x in range(self.n_x):
            for y in range(self.n_y):
                self.approximated_phi_vertical_derivative[x, y] = \
                    ((self.approximated_phi[x, (y + 1) % self.n_y] - self.approximated_phi[x, y])*self.n_y) # phi is cyclic for T=1

    def _approximate_phi_horizontal_derivative(self):
        self.approximated_phi_horizontal_derivative = np.zeros((self.n_x, self.n_y))
        for y in range(self.n_y):
            for x in range(self.n_x):
                self.approximated_phi_horizontal_derivative[x, y] = \
                    ((self.approximated_phi[(x + 1) % self.n_x, y] - self.approximated_phi[x, y])*self.n_x) # phi is cyclic for T=1

    def approximate_vertical_derivative_energy(self):
        self._approximate_phi_vertical_derivative()
        self.approximated_vertical_derivative_energy = np.sum(self.approximated_phi_vertical_derivative**2)/(self.n_x * self.n_y)

    def approximate_horizontal_derivative_energy(self):
        self._approximate_phi_horizontal_derivative()
        self.approximated_horizontal_derivative_energy = np.sum(self.approximated_phi_horizontal_derivative**2)/(self.n_x * self.n_y)

def run_sections(A: float, w_x: float, w_y: float, hor_samples: int, ver_samples: int):
    phi = Phi(A, w_x, w_y)
    aprox_phi = ApproximatePhi(phi, hor_samples, ver_samples)
    # aprox_img = aprox_phi.approximated_phi
    # plt.imshow(aprox_img, 'gray', vmin = -phi.A, vmax = phi.A)
    # plt.title("Approximated phi for {}*{} samples".format(aprox_phi.hor_samples, aprox_phi.ver_samples))
    # plt.show()

    aprox_phi.approximate_vertical_derivative_energy()
    print("approximated vertical derivative energy: {}".format(aprox_phi.approximated_vertical_derivative_energy))

    aprox_phi.approximate_horizontal_derivative_energy()
    print("approximated horizontal derivative energy: {}".format(aprox_phi.approximated_horizontal_derivative_energy))

    aprox_phi.approximate_val_range()
    print("approximated value range: {}".format(aprox_phi.val_range))


if __name__ == '__main__':
    run_sections(2500, 2, 7, 200, 700)