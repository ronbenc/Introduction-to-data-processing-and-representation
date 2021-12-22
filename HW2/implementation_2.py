import math
import numpy as np

class Hadamard:
    def __init__(self, n: int) -> None:
        self.n = n
        if n == 0:
            self.matrix = np.matrix([1])

        elif n == 1:
            self.matrix = (1/math.sqrt(2))*np.matrix([[1, 1], [1, -1]])
        
        else:
            size = 2**n
            prev_matrix = Hadamard(n-1).matrix
            self.matrix = np.empty((size, size))
            self.matrix[0: int(size/2), 0: int(size/2)] = prev_matrix
            self.matrix[0: int(size/2), int(size/2): size] = prev_matrix
            self.matrix[int(size/2): size, 0: int(size/2)] = prev_matrix
            self.matrix[int(size/2): size, int(size/2): size] = -prev_matrix
            self.matrix *= (1/math.sqrt(size))



if __name__ == '__main__':
    mat1 = Hadamard(0)
    print(mat1.matrix)
    mat2 = Hadamard(1)
    print(mat2.matrix)
    mat8 = Hadamard(3)
    print(mat8.matrix)