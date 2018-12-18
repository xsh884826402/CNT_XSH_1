import numpy as np
def batch_iter(x, y, batch_size):
    data_len = len(x)
    num_batch = int(data_len/batch_size)+1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
def batch_iter_1(x, batch_size):
    data_len = len(x)
    num_batch = int(data_len/batch_size)+1

    for i in range(num_batch):
        start_id = i*batch_size
        end_id =min((i+1)*batch_size, data_len)
        yield x[start_id:end_id]
