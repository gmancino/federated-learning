import numpy as np
import math
import torch
import torchvision
import matplotlib.pyplot as plt
from MulticlassLogReg import MulticlassLogReg
from CreateAgents import CreateAgents

# Import Data using pytorch
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_data = torchvision.datasets.CIFAR10('../../../../../Data/', transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data))

# Load data into numpy arrays
x_data = next(iter(train_loader))[0].numpy()
y_data = next(iter(train_loader))[1].numpy()
train_label = y_data[0:40000]
test_label = y_data[40000:50000]
train_data = np.zeros((40000, 1024))
test_data = np.zeros((10000, 1024))

for i in range(40000):
    # Convert to grayscale
    train_data[i, :] = 0.3 * x_data[i][0].flatten() + 0.59 * x_data[i][1].flatten() + 0.11 * x_data[i][2].flatten()

for i in range(10000):
    test_data[i, :] = 0.3 * x_data[40000 + i][0].flatten() + 0.59 * x_data[40000 + i][1].flatten() + 0.11 * x_data[40000 + i][2].flatten()

# Run multiclass logistic regression to get "w_true"
all_data_train = MulticlassLogReg(train_data, train_label, np.random.rand(train_data.shape[1], 10), np.random.rand(10), T=1000, tol=0.5, alpha=1e-1, lam1=1e-2, lam2=1, num_classes=10)
all_data_train.optimization()

pred, prob = all_data_train.predict(test_data, all_data_train.w, all_data_train.b)

print('Accuracy: ', sum(pred == test_label)/len(pred))

# Save "true" values for comparison
w_true = all_data_train.w
b_true = all_data_train.b
b_true_mean = b_true - np.mean(b_true)

w_true = np.concatenate((w_true, np.array([b_true_mean])), axis=0)

w_seq = []
corruptions_seq = []

for i in range(1, 16):

    y_corrupt_ind = np.random.randint(len(test_label[0:100]), size=math.floor((i * 0.01) * len(test_label[0:1000])))
    corruptions = np.random.randint(10, size=len(y_corrupt_ind))
    y_corrupt = test_label[0:100].copy()
    y_corrupt[y_corrupt_ind] = corruptions

    corruption_log_reg = MulticlassLogReg(test_data[0:100], y_corrupt, np.random.rand(test_data.shape[1], 10), np.random.rand(10), T=1000, tol=0.5, alpha=0.25, lam1=1e-3, lam2=1, num_classes=10)
    corruption_log_reg.optimization()

    w = np.concatenate((corruption_log_reg.w, np.array([corruption_log_reg.b - np.mean(b_true)])), axis=0)

    norm = np.linalg.norm(w - w_true, ord='fro')/np.linalg.norm(w_true, ord='fro')

    corruptions_seq.append(i * 0.01)
    w_seq.append(norm)


fig = plt.figure()
plt.plot(corruptions_seq, w_seq)
plt.title(str(len(test_label[0:100])) + ' data points')
plt.xlabel('Percent corruptions')
plt.ylabel('||w_c-w_true||_F/||w_true||_F')