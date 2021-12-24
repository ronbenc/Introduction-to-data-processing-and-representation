from abc import abstractmethod
import math
import numpy as np
from matplotlib import pyplot as plt

class Matrix:
    def __init__(self, n: int) -> None:
        self.n = n
        self.size = 2**n
        self.matrix = np.empty((self.size, self.size))
    
    def plot_columns(self):
        x_axis = [x/self.size for x in range(0, self.size+1)]
        fig,axs = plt.subplots(self.size)
        tmp_mat = np.vstack((self.matrix[0, :], self.matrix)) #TODO resove this more elegantly
        for i in range(self.size):
            axs[i].step(x_axis, tmp_mat[:, i]) #how to plot with bars
        
        plt.show()
        
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
            upper_haar = (1/np.sqrt(2))*np.kron(prev_haar.matrix, np.array([1, 1]))
            lower_haar = (1/np.sqrt(2))*np.kron(np.eye(prev_haar.size), np.array([1, -1]))
            self.matrix = np.vstack((upper_haar, lower_haar)).T

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
        self._calc_std_coefficients()
        self._calc_all_coefficients()
        self._sort_all_coefficients()

    def approximate(self, k: int) -> None:
        self.k_coeff = np.empty((k))
        self._choose_k_coefficients(k)

    def calc_MSE(self) -> float:
        coefficients_error = np.sum(np.power(self.k_coeff, 2))
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
    
    def _sort_all_coefficients(self):
        self.sorted_coeff = sorted(self.coeff, key=abs, reverse=True)

    def _choose_k_coefficients(self, k: int):
        self.k_coeff = self.sorted_coeff[:k]


def create_basis(matrix: Matrix, sum_range: float, N: int):
    return matrix.matrix.T*np.sqrt(N/sum_range)




if __name__ == '__main__':
    hadamard = Hadamard(2)
    hadamard.plot_columns()
    print(hadamard.matrix)
    # walsh_hadamard = WalshHadamard(hadamard)
    # walsh_hadamard.plot_columns()
    # print(walsh_hadamard.matrix)
    haar = Haar(2)
    # haar.plot_columns()
    phi = Phi((-4, 5))
    phi_pre = PhiPre((-4, 5))
    basis = create_basis(Hadamard(2), phi.range[1]-phi.range[0], 4)
    k_term_aprox = KTermApproximator(phi, phi_pre, basis, 4)
    for k in range(1, 5):
        k_term_aprox.approximate(k)
        print(k_term_aprox.k_coeff)
        print(k_term_aprox.calc_MSE())