import numpy as np



class SGD:

    def __init__(self, model, lr=0.01, momentum=0.0):
        self.l = model.layers
        self.L = len(model.layers)
        self.mu = momentum
        self.lr = lr
        self.v_W = [np.zeros(self.l[i].W.shape) for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape) for i in range(self.L)]


    def step(self):
        for i in range(self.L):
            if self.mu == 0:
                self.l[i].W -= self.lr * self.l[i].dLdW
                self.l[i].b -= self.lr * self.l[i].dLdb
            else:
                self.v_W[i] = self.mu * self.v_W[i] + self.l[i].dLdW
                self.v_b[i] = self.mu * self.v_b[i] + self.l[i].dLdb
                self.l[i].W -= self.lr * self.v_W[i]
                self.l[i].b -= self.lr * self.v_b[i]
