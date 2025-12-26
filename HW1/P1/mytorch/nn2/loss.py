import numpy as np
from mytorch.nn2.activation import Softmax
class MSELoss:

    def forward(self, A, Y ):
        self.A = A
        self.Y = Y

        self.N = A.shape[0]
        self.C = A.shape[1]

        SE = (A-Y)*(A-Y)

        SSE = np.ones((self.N,1)).T @ SE @ np.ones((self.C,1))

        return SSE / (self.N * self.C)

    def backward(self):

        dLdA = 2*(self.A-self.Y)/(self.N * self.C)
        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        Ones_C = np.ones((self.C,1))
        Ones_N = np.ones((self.N,1))

        A_shift = A - np.max(A, axis=1, keepdims=True)
        exp_A = np.exp(A_shift)

        self.softmax = exp_A / np.sum(exp_A, axis=1).reshape(-1,1)
        crossentropy = (-Y * np.log(self.softmax)) @ np.ones((self.C, 1))

        sum_crossentropy_loss = np.ones((self.N, 1)).T @ crossentropy

        mean_crossentropy_loss = sum_crossentropy_loss/self.N
        return mean_crossentropy_loss

    def backward(self):
        return (self.softmax - self.Y) / self.N
