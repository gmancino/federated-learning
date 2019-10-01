import numpy as np


class MulticlassLogReg():

    def __init__(self, x, y, w0, b0, alpha, T, tol, lam1, lam2, num_classes):
        self.x = x
        self.y = y
        self.w0 = w0
        self.b0 = b0
        self.w = self.w0.copy()
        self.b = self.b0.copy()
        self.stepsize = alpha
        self.max_iters = T
        self.tol = tol
        self.lambdas = [lam1, lam2]
        self.num_classes = num_classes
        self.losses = []

    def softmax(self, z):
        # Subtract max for numerical stability
        z -= np.max(z)
        sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
        return sm

    def predict(self, x, w, b):
        probability = self.softmax(np.dot(x, w) + b)
        prediction = np.argmax(probability, axis=1)
        return [prediction, probability]

    def one_hot_encoding(self, y, num_classes):
        ln = len(y)
        y = y.astype(int)
        Y = np.zeros((ln, num_classes))
        for i in range(ln):
            Y[i, y[i]] = 1
        return Y

    def grad(self, x, y, w, b, lam1, lam2, num_classes):

        # Encode data into matrix
        yhat = self.one_hot_encoding(y, num_classes)

        N = x.shape[0]

        scores = np.dot(x, w) + b
        probs = self.softmax(scores)

        # Frobenius norm loss
        loss = (-1/N) * np.sum(yhat * np.log(probs)) + (lam1/2)*np.linalg.norm(w, ord='fro')**2 + (lam2/2) * np.linalg.norm(b, ord='fro')**2

        gradw = (-1/N) * np.dot(x.T,(yhat - probs)) + lam1*w
        gradb = (-1/N) * (yhat - probs) + lam2*b

        return [gradw, gradb, loss]

    def optimization(self):

        for i in range(self.max_iters):
            [gradw, gradb, loss] = self.grad(self.x, self.y, self.w, self.b, self.lambdas[0], self.lambdas[1], self.num_classes)

            self.losses.append(loss)

            self.w = self.w - self.stepsize * gradw
            self.b = self.b - self.stepsize * gradb

        self.b = self.b[0, :]

        return [self.losses, self.w, self.b]
