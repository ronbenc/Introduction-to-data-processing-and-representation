import math
import numpy as np
from matplotlib import pyplot as plt

class Hadamard:
    def __init__(self, n: int) -> None:
        self.n = n
        self.self = 2**n
        if n == 0:
            self.matrix = np.matrix([1])

        elif n == 1:
            self.matrix = (1/math.sqrt(2))*np.matrix([[1, 1], [1, -1]]) #should we normalize?
        
        else:
            self.size = 2**n
            prev_matrix = Hadamard(n-1).matrix
            self.matrix = np.empty((self.size, self.size))
            self.matrix[0: int(self.size/2), 0: int(self.size/2)] = prev_matrix
            self.matrix[0: int(self.size/2), int(self.size/2): self.size] = prev_matrix
            self.matrix[int(self.size/2): self.size, 0: int(self.size/2)] = prev_matrix
            self.matrix[int(self.size/2): self.size, int(self.size/2): self.size] = -prev_matrix
            self.matrix *= (1/math.sqrt(2)) #should we normalize?

class WalshHadamard:
    def __init__(self, hadamard: Hadamard) -> None:
        self.size = hadamard.size
        self.hadamard_matrix = hadamard.matrix
        self._get_changes_of_sign_dict()
        self._sort_by_changes_of_sign()

    @staticmethod
    def _changes_of_sign(arr: np.array):
        count = 0
        for i in range(len(arr)-1):
            if (arr[i] > 0 and arr[i+1] < 0) or (arr[i] < 0 and arr[i+1] > 0):
                count+=1

        return count

    def _get_changes_of_sign_dict(self):
        self._changes_of_sign_dict = {} # a mapping between number of changes in a row and the row index
        for row in range(self.size):
            curr_num_of_sign_changes = self._changes_of_sign(self.hadamard_matrix[row, :])
            self._changes_of_sign_dict[curr_num_of_sign_changes] = row

    #TODO improve for linear time sorting
    def _sort_by_changes_of_sign(self):
        self.matrix = np.empty((self.size, self.size))
        for key, index in self._changes_of_sign_dict.items():
            self.matrix[key, :] = self.hadamard_matrix[index, :]


def plot_h(mat):
    x_axis = [x/mat.size for x in range(0, mat.size)]
    fig,axs = plt.subplots(mat.size)
    for i in range(mat.size):
        axs[i].plot(x_axis, mat.matrix[:, i]) #how to plot with bars
        
    plt.show()

if __name__ == '__main__':
    hadamard = Hadamard(3)
    # plot_h(hadamard)
    walsh_hadamard = WalshHadamard(hadamard)
    plot_h(walsh_hadamard)