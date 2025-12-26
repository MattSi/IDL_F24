import numpy as np
import scipy as sp

class Identity:
    def forward(self, Z):
        self.Z = Z
        self.A = Z
        return self.A
    def backward(self, dLdA):
        dAdZ = np.ones_like(self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dAdz = (self.A - self.A * self.A)
        dLdZ = dLdA * dAdz

        return dLdZ


class Tanh:
    def forward(self, Z):
        self.A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
        self.expZ = np.exp(Z)
        self.expmZ = np.exp(-Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = 4 / (2+self.expZ*self.expZ + self.expmZ*self.expmZ)
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        dLdZ = dLdA * (self.A>0)
        return dLdZ


class GELU:
    def forward(self, Z):
        self.Z = Z
        self.A = 0.5*Z*(1 + sp.special.erf(Z/np.sqrt(2)) )
        return self.A

    def backward(self, dLdA):
        dAdZ = 0.5*(1+sp.special.erf(self.Z/np.sqrt(2)) ) + self.Z/(np.sqrt(2* np.pi)) * np.exp(-self.Z*self.Z/2)
        dLdZ = dLdA * dAdZ
        return dLdZ


class Softmax:
    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shift)
        self.A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.A


    def backward(self, dLdA):
        # Calculate the batch size and number of features
        N = dLdA.shape[0]
        C = dLdA.shape[1]

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N, C))

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i, :] = np.dot(dLdA[i, :], J)

        return dLdZ

