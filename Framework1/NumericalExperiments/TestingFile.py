
# Find "true" parameters by training on a majority of the data
[w_true, loss_hist] = multiclass_log_reg(x[0:50000, :], y[0:50000], 10, 0.25, 1, 1000)

# Train on just a subset to see how w changes based on number of data points
[w_i, loss_hist_i] = multiclass_log_reg(x_train_agents[0], y_train_agents[:, 0], 10, 0.25, 1, 1000)
print(la.norm(w_i - w_true, ord='fro'))

#y_corrupt = np.random.randint(10, size=len(y_train_agents[:, 0]))
#y_corrupt = np.zeros(len(y_train_agents))
y_corrupt_ind = np.random.randint(len(y_test), size=math.floor(len(y_test)))
corruptions = np.random.randint(10, size=len(y_corrupt_ind))
y_corrupt = y_test.copy()
y_corrupt[y_corrupt_ind] = corruptions

[w_c, loss_hist_c] = multiclass_log_reg(x_test, y_corrupt, 10, 0.25, 1, 1000)
print(la.norm(w_c - w_true, ord='fro')/la.norm(w_true, ord='fro'))

w_seq = []
corruptions_seq = []

for i in range(1, 16):

    y_corrupt_ind = np.random.randint(len(y_test[0:1000]), size=math.floor((i * 0.01) * len(y_test[0:1000])))
    corruptions = np.random.randint(10, size=len(y_corrupt_ind))
    y_corrupt = y_test[0:1000].copy()
    y_corrupt[y_corrupt_ind] = corruptions

    [w, l] = multiclass_log_reg(x_test[0:1000], y_corrupt, 10, 0.25, 1, 1000)

    norm = la.norm(w - w_true, ord='fro')/la.norm(w_true, ord='fro')

    corruptions_seq.append(i * 0.01)
    w_seq.append(norm)


fig = plt.figure()
plt.plot(corruptions_seq, w_seq)
plt.title(str(len(y_test[0:1000])) + ' data points')
plt.xlabel('Percent corruptions')
plt.ylabel('||w_c-w_true||_F/||w_true||_F')