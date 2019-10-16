# ----------------------------------
# Kernelization of input data
#
#
# Author: Gabe Mancino
# ----------------------------------

# Import necessary packages

import math
import numpy as np

class Kernelize:

    def __init__(self, x, deg, shift, proj_dim, final_dim):

        self.data = x
        self.deg = deg
        self.shift = shift
        self.proj_dim = proj_dim
        self.final_dim = final_dim

    def poly_kern(self):

        print('[INFO] Kernelization beginning...\n[INFO] Performing degree ', self.deg, ' polynomial kernelization with bias ', self.shift, '.')

        [n, d] = self.data.shape

        if self.proj_dim < d:
            print('[INFO] Project into a higher dimension than the data lies in.\n')
            return

        # Make a bias vector of dimension (1 x n)
        shift = (1 / math.sqrt(self.shift)) * np.array([np.ones(n)])

        # Concatenate to original matrix before undergoing transformation
        self.data = np.concatenate((self.data, shift.T), axis=1)

        # Compute approximate kernel mapping
        for i in range(self.deg):

            print('[INFO] Working on iteration ', i, ' for Kernelization...')

            if i == 0:
                # Generate random sign matrix
                w1 = np.random.randint(0, 2, size=(self.proj_dim, d + 1))
                w2 = w1 - 1
                w = w1 + w2

                # Compute mapping
                w_data = np.dot(w, np.transpose(self.data))


            else:
                w_old = w_data

                # Generate random sign matrix
                w1 = np.random.randint(0, 2, size=(self.proj_dim, d + 1))
                w2 = w1 - 1
                w = w1 + w2

                # Compute mapping
                w_data =  np.dot(w, np.transpose(self.data))

                w_data = np.multiply(w_data, w_old)

        print('[INFO] Projecting into lower dimensional space by using the JLT lemma...')

        r = (1 / math.sqrt(self.final_dim)) * np.random.normal(0, 1, size=(self.final_dim, w_data.shape[0]))

        self.kernel = np.dot(r, (1 / math.sqrt(self.proj_dim)) * w_data).T

        print('[INFO] Kernelization complete.')

        return self.kernel