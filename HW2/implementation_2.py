import math
import numpy as np
from matplotlib import pyplot as plt

class Matrix:
    def __init__(self, n: int) -> None:
        self.n = n
        self.size = 2**n
        self.matrix = np.empty((self.size, self.size))
    
    def plot_columns(self):
        x_axis = [x/self.size for x in range(0, self.size)]
        fig,axs = plt.subplots(self.size)
        for i in range(self.size):
            axs[i].step(x_axis, self.matrix[:, i]) #how to plot with bars
        
        plt.show()
        
class Hadamard(Matrix):
    def __init__(self, n: int) -> None:
        super().__init__(n)
       
        if n == 0:
            self.matrix = np.matrix([1])

        elif n == 1: 
            self.matrix = (1/math.sqrt(2))*np.matrix([[1, 1], [1, -1]]) #should we normalize?
        
        else:
            prev_matrix = Hadamard(n-1).matrix
            self.matrix[0: int(self.size/2), 0: int(self.size/2)] = prev_matrix
            self.matrix[0: int(self.size/2), int(self.size/2): self.size] = prev_matrix
            self.matrix[int(self.size/2): self.size, 0: int(self.size/2)] = prev_matrix
            self.matrix[int(self.size/2): self.size, int(self.size/2): self.size] = -prev_matrix
            self.matrix *= (1/math.sqrt(2)) #should we normalize?

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
            self.matrix = np.matrix([1])

        else:
            prev_haar = Haar(n-1)
            upper_haar = (1/math.sqrt(2))*np.kron(prev_haar.matrix, np.array([1, 1]))
            lower_haar = (1/math.sqrt(2))*np.kron(np.identity(prev_haar.size), np.array([1, -1]))
            self.matrix = np.vstack((upper_haar, lower_haar))
    


if __name__ == '__main__':
    # hadamard = Hadamard(2)
    # hadamard.plot_columns()
    # walsh_hadamard = WalshHadamard(hadamard)
    # walsh_hadamard.plot_columns()
    haar = Haar(2)
    print(haar.matrix)