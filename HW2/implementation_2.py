from abc import abstractmethod
import math
from typing import List
import numpy as np
from matplotlib import pyplot as plt

class Matrix:
    def __init__(self, n: int) -> None:
        self.n = n
        self.size = 2**n
        self.matrix = np.empty((self.size, self.size))
        
class Hadamard(Matrix):
    def __init__(self, n: int) -> None:
        super().__init__(n)
       
        if n == 0:
            self.matrix = np.eye(1)

        elif n == 1: 
            self.matrix = (1/math.sqrt(2))*np.array([[1, 1], [1, -1]]) #should we normalize?
        
        else:
            prev_matrix = Hadamard(n-1).matrix
            Hadamard2matrix = Hadamard(1).matrix
            self.matrix = np.kron(Hadamard2matrix, prev_matrix)

class Eye(Matrix):
    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.matrix = np.eye(self.size)

class WalshHadamard(Matrix):
    def __init__(self, hadamard: Hadamard) -> None:
        super().__init__(hadamard.n)

        self.hadamard = hadamard
        self._get_changes_of_sign_dict()
        self._sort_by_changes_of_sign()

    @staticmethod
    def _count_changes_of_sign(arr: np.array):
        count = 0
        for i in range(len(arr)-1):
            if (arr[i] > 0 and arr[i+1] < 0) or (arr[i] < 0 and arr[i+1] > 0):
                count+=1

        return count

    def _get_changes_of_sign_dict(self):
        self._changes_of_sign_dict = {} # a mapping between number of changes in a row and the row index
        for row in range(self.size):
            curr_num_of_sign_changes = self._count_changes_of_sign(self.hadamard.matrix[row, :])
            self._changes_of_sign_dict[curr_num_of_sign_changes] = row

    #TODO improve for linear time sorting
    def _sort_by_changes_of_sign(self):
        for key, index in self._changes_of_sign_dict.items():
            self.matrix[key, :] = self.hadamard.matrix[index, :]

class Haar(Matrix):
    def __init__(self, n: int) -> None:
        super().__init__(n)
        if n == 0:
            self.matrix = np.eye(1)

        else:
            prev_haar = Haar(n-1)
            left_haar = (1/np.sqrt(2))*np.kron(prev_haar.matrix, np.array(([1],[1])))
            right_haar = (1/np.sqrt(2))*np.kron(np.eye(prev_haar.size), np.array(([1],[-1])))
            self.matrix = np.hstack((left_haar, right_haar))

class Func:
    def __init__(self, range) -> None:
        self.range = range
        self.integral_error = None

    @abstractmethod
    def compute(self, t: float) -> float:
        ...

class Phi(Func):
    def __init__(self, range: float) -> None:
        super().__init__((range))
        self.integral_error = 2.257712709582769*(10**5) #calculated analyticly

    def compute(self, t: float) -> float:
        return t*(np.exp(t))

class PhiPre(Func):
    def __init__(self, range: float) -> None:
        super().__init__(range)

    def compute(self, t: float) -> float:
        return (t-1)*np.exp(t)


class KTermApproximator:
    def __init__(self, phi: Func, phi_predecesor: Func, basis, N: int) -> None:
        self.phi = phi
        self.phi_predecesor = phi_predecesor
        self.basis = basis
        self.N = N
        self.std_basis_coeff = np.empty((N))
        self.coeff = np.empty((N))
        self.sorted_coeff = np.empty((N))
        self.k_coeff = None
        self.reconstructed_phi = None
        self._calc_std_coefficients()
        self._calc_all_coefficients()
        self._sort_all_coefficients()

    def approximate(self, k: int) -> None:
        self.k_coeff = np.empty((k))
        self._choose_k_coefficients(k)

    def calc_MSE(self) -> float:
        clean_k_coeff = [coeff for _,coeff in self.k_coeff]
        coefficients_error = np.sum(np.power(clean_k_coeff, 2))
        return self.phi.integral_error - coefficients_error

    def _calc_std_coefficients(self):
        interval_size = (self.phi.range[1] - self.phi.range[0])/self.N
        for coeff_index,range_start in enumerate(np.linspace(self.phi.range[0], self.phi.range[1], self.N+1)):
            if coeff_index == self.N:
                break
            self.std_basis_coeff[coeff_index] = self.phi_predecesor.compute(range_start + interval_size) - self.phi_predecesor.compute(range_start)

        self.std_basis_coeff = self.std_basis_coeff

    def _calc_all_coefficients(self):
        self.coeff = np.matmul(self.basis, self.std_basis_coeff.T)
        self.coeff = [i for i in enumerate(self.coeff)]
    
    def _sort_all_coefficients(self):
        f = lambda x : abs(x[1])
        self.sorted_coeff = sorted(self.coeff, key=f, reverse=True)

    def _choose_k_coefficients(self, k: int):
        self.k_coeff = self.sorted_coeff[:k]

    def reconstruct_phi(self):
        self.reconstructed_phi = np.zeros(self.N)

        for index, coeff in self.k_coeff:
            self.reconstructed_phi += coeff*self.basis[index,: ]


def create_basis(matrix: Matrix, sum_range: float, N: int):
    return matrix.matrix.T*np.sqrt(N/sum_range)

def plot_columns(matrix, N: int, mat_type = None):
    x_axis = [x/N for x in range(0, N+1)]
    if N <= 8:
        fig,axs = plt.subplots(N)
        tmp_mat = np.vstack((matrix[0, :], matrix)) #TODO solve this more elegantly
        for i in range(N):
            axs[i].step(x_axis, tmp_mat[:, i])
    
    else:
        k = int(N/8)
        fig,axs = plt.subplots(int(N/k), k)
        tmp_mat = np.vstack((matrix[0, :], matrix)) #TODO solve this more elegantly
        for i in range(N):
            axs[int(i/k), i%k].step(x_axis, tmp_mat[:, i])

    
    if mat_type == None:
        plt.suptitle("n = {}".format(int(np.log2(N))))

    else:
        plt.suptitle("{}, n = {}".format(mat_type, int(np.log2(N))))

    plt.show()

def plot_MSE(k_term_aprox: KTermApproximator, ks: list, basis_name):
    mse_list = []
    for k in ks:
        k_term_aprox.approximate(k)

        mse_list.append(k_term_aprox.calc_MSE())

    print(mse_list)

    plt.plot(ks, mse_list)
    plt.title(basis_name)
    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.show()

def plot_recontructed_phi(k_term_aprox: KTermApproximator, ks: list, basis_name):
    x_axis = np.linspace(k_term_aprox.phi.range[0], k_term_aprox.phi.range[1], 5)
    for k in ks:
        k_term_aprox.approximate(k)
        k_term_aprox.reconstruct_phi()
        reconstructed_phi = [k_term_aprox.reconstructed_phi[0]]
        reconstructed_phi.extend(k_term_aprox.reconstructed_phi)
        print(reconstructed_phi)
        plt.step(x_axis, reconstructed_phi,where='pre')
        plt.title(basis_name + ", k = {}".format(k))
        plt.show()
        


if __name__ == '__main__':
    # # plot hadamards
    # for n in [2, 3, 4, 5, 6]:
    #     hadamard = Hadamard(n)
    #     N = hadamard.size
    #     basis = create_basis(hadamard, 1, N)
    #     plot_columns(basis, N, "Hadamard")

    # # plot walsh-hadamards
    # for n in [2, 3, 4, 5, 6]:
    #     hadamard = Hadamard(n)
    #     walsh_hadamard = WalshHadamard(hadamard)
    #     N = walsh_hadamard.size
    #     basis = create_basis(walsh_hadamard, 1, N)
    #     plot_columns(basis, N, "Walsh-Hadamard")


    # # plot haars
    # for n in [2, 3, 4, 5, 6]:
    #     haar = Haar(n)
    #     haar.matrix = haar.matrix.T
    #     N = haar.size
    #     basis = create_basis(haar, 1, N)
    #     plot_columns(basis, N, "Haar")


    phi = Phi((-4, 5))
    phi_pre = PhiPre((-4, 5))
    ks = [1, 2, 3, 4]
    func_range = 9

    # eye = Eye(2)
    # standard_basis = create_basis(eye, func_range, 4)
    # k_term_aprox = KTermApproximator(phi, phi_pre, standard_basis, 4)
    # k_term_aprox.approximate(4)
    # k_term_aprox.reconstruct_phi()
    # plot_recontructed_phi(k_term_aprox, ks, "Standard basis")
    # plot_MSE(k_term_aprox, ks, "Standard basis")

    # hadamard = Hadamard(2)
    # hadamarad_basis = create_basis(hadamard, func_range, 4)
    # k_term_aprox = KTermApproximator(phi, phi_pre, hadamarad_basis, 4)
    # k_term_aprox.approximate(4)
    # k_term_aprox.reconstruct_phi()
    # plot_recontructed_phi(k_term_aprox, ks, "Hadamard basis")
    # plot_MSE(k_term_aprox, ks, "Hadamard basis")

    # hadamard = Hadamard(2)
    # walsh_hadamard = WalshHadamard(hadamard)
    # walsh_hadamard_basis = create_basis(walsh_hadamard, func_range, 4)
    # k_term_aprox = KTermApproximator(phi, phi_pre, walsh_hadamard_basis, 4)
    # k_term_aprox.approximate(4)
    # k_term_aprox.reconstruct_phi()
    # plot_recontructed_phi(k_term_aprox, ks, "Walsh-Hadamard basis")
    # plot_MSE(k_term_aprox, ks, "Walsh-Hadamard basis")

    haar = Haar(2)
    haar_basis = create_basis(haar, func_range, 4)
    k_term_aprox = KTermApproximator(phi, phi_pre, haar_basis, 4)
    k_term_aprox.approximate(4)
    k_term_aprox.reconstruct_phi()
    plot_recontructed_phi(k_term_aprox, ks, "haar basis")
    plot_MSE(k_term_aprox, ks, "haar")