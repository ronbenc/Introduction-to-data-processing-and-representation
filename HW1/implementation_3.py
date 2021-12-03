from matplotlib import pyplot as plt
import numpy as np

class IRLS:
    def __init__(self, f, N, epsilon, p = 1) -> None:
        self.f = f
        self.N = N
        self.epsilon = epsilon
        self.p = p
        self.w = np.eye(f.shape)
        self.g = np.zeros_like(f) # f^
        self.I = f.shape[0]/N
        self.error = None
    

    def workflow(self):
        while(True):
            self.g = self.get_updated_g()
            self.w = self.get_updated_w()
            prev_error = self.error
            self.error = self.calc_error()
            if(prev_error is not None and (prev_error-self.error) < self.delta):
                break

    def get_updated_g(self):
        new_g = np.zeros_like(self.g)

        for row in range(self.N):
            for col in range(self.N):
                new_g[row*self.I:(row*self.I+1),col*self.I:col(*self.I+1)] = self.get_updated_g_i(row, col)
        
        return new_g

    def get_updated_g_i(self, row, col) -> float:
        w_i = self.w[row*self.I:(row*self.I+1),col*self.I:col(*self.I+1)]
        f_i = self.f[row*self.I:(row*self.I+1),col*self.I:col(*self.I+1)]

        return (w_i * f_i).sum()/w_i.sum()

    def get_updated_w(self):
        normalized_w = (self.f - self.g)**(self.p-2)
        normalized_w[normalized_w > (1/self.epsilon)] = 1/self.epsilon
        return normalized_w

    def calc_error(self):
        ...

if __name__ == '__main__':
    ...