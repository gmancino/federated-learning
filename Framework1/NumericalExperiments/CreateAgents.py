# Import packages

import math
import numpy as np
from sklearn.model_selection import train_test_split


# Define class to divide data among agents
class CreateAgents():

    def __init__(self, x, y, k, percent_test):
        self.x = x
        self.y = y
        self.k = k
        self.percent_test = percent_test

    def create_agents(self):

        d = self.x.shape[1]

        # Save percent_test% of data points for testing
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.percent_test, random_state=10)

        # Get number of training points
        m = x_train.shape[0]

        if m % math.floor(m / self.k) == 0:
            print('[INFO] Distributing data to ' + str(self.k) + ' \'agents\'...')

            # Allocate space
            num = math.floor(m / self.k)
            x_train_agents = np.zeros((self.k, num, d))
            y_train_agents = np.zeros((num, self.k))

            # Chop up data into different clusters
            for i in range(self.k):
                if i == 0:
                    x_train_agents[i, :, :] = x_train[0:num, :]
                    y_train_agents[:, i] = y_train[0:num]
                elif (i < self.k - 1):
                    x_train_agents[i, :, :] = x_train[(i * num):((i + 1) * num), :]
                    y_train_agents[:, i] = y_train[(i * num):((i + 1) * num)]
                else:
                    x_train_agents[i, :, :] = x_train[(i * num):m, :]
                    y_train_agents[:, i] = y_train[(i * num):m]

            print('[INFO] Done sorting...')

            return [x_train_agents, y_train_agents, x_test, y_test]

        else:
            print('[INFO] Uneven number of data points to distribute.')

