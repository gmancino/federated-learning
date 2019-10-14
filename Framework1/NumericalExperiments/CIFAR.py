import numpy as np
import math
import torch
import torchvision
import matplotlib.pyplot as plt
#from MulticlassLogReg import MulticlassLogReg
#from CreateAgents import CreateAgents
from sklearn.linear_model import LogisticRegression

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

sk = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=2000).fit(train_data, train_label)
w_true = sk.coef_
b_true = sk.intercept_

w_true = np.concatenate((w_true.T, np.array([b_true])))

#plt.imshow(train_data[0, :].reshape(32, 32), cmap=plt.get_cmap('gray'), interpolation='none')

#w0 = np.random.rand(train_data.shape[1], 10)
#b0 = np.random.rand(10)

# Run multiclass logistic regression to get "w_true"
#all_data_train = MulticlassLogReg(train_data, train_label, w0, b0, T=1000, tol=0.5, alpha=1e-2, lam1=1e-1, lam2=1e-1, num_classes=10)
#all_data_train.grad_descent()
#pred, prob = all_data_train.predict(test_data, all_data_train.w, all_data_train.b)
#print('Accuracy: ', sum(pred == test_label)/len(pred))

# Save "true" values for comparison
#w_true = all_data_train.w
#b_true = all_data_train.b
#b_true_mean = b_true - np.mean(b_true)

#w_true = np.concatenate((w_true, np.array([b_true_mean])), axis=0)

w_seq = []
corruptions_seq = []
accuracy = []
num_test = 10000

for i in range(1, 11):

    y_corrupt_ind = np.random.randint(len(test_label[0:num_test]), size=math.floor((i * 0.01) * len(test_label[0:num_test])))
    corruptions = np.random.randint(10, size=len(y_corrupt_ind))
    y_corrupt = test_label[0:num_test].copy()
    y_corrupt[y_corrupt_ind] = corruptions

    sk = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=2000).fit(test_data[0:num_test], y_corrupt)
    w = sk.coef_
    b = sk.intercept_

    w = np.concatenate((w.T, np.array([b])))

    norm = np.linalg.norm(w - w_true, ord='fro')/np.linalg.norm(w_true, ord='fro')

    pred = sk.predict(test_data[0:num_test])

    acc = sum(pred == test_label[0:num_test]) / len(pred)

    corruptions_seq.append(i * 0.01)
    w_seq.append(norm)
    accuracy.append(acc)


plt.subplot(211)
plt.plot(corruptions_seq, w_seq)
plt.title(str(len(test_label[0:num_test])) + ' data points on CIFAR-10 data')
plt.ylabel('||w_c-w_true||_F/||w_true||_F')
plt.subplot(212)
plt.plot(corruptions_seq, accuracy)
plt.ylabel('Classification accuracy')
plt.xlabel('Corruption percentages')
plt.show()