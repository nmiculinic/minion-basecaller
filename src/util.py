import tensorflow as tf
import numpy as np
import os
import h5py
import random


def atrous_conv1d(value, filters, rate, padding="SAME", name=None):
    with tf.name_scope(name, "atrous_conv1d", [value, filters]) as name:
        value = tf.convert_to_tensor(value, name="value")
        filters = tf.convert_to_tensor(filters, name="filters")

        if rate == 1:
            return tf.nn.conv1d(value, filters, 1, padding)

        if value.get_shape().is_fully_defined():
            value_shape = value.get_shape().as_list()
        else:
            value_shape = tf.shape(value)

        add = (-value_shape[1] % rate + rate) % rate
        pad = [[0, add]]
        crop = [[0, add]]

        value = tf.space_to_batch_nd(input=value,
                                     paddings=pad,
                                     block_shape=[rate])

        value = tf.nn.conv1d(value, filters, 1, padding, name=name)

        value = tf.batch_to_space_nd(input=value,
                                     crops=crop,
                                     block_shape=[rate])

        return value


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def encode(arr):
    """Create a sparse representention of x.
        Adjacent repetative symbols are encoded with difference 4 (e.g. AAA -> 0,4,0). Blanks are removed (N)
    Args:
        arr: 2D unicode matrix with AGTC or N charaters
    Returns:
        y_out: label encoded matrix.
        poss: label string length
    """
    y_out = np.zeros_like(arr, dtype=np.int32)
    poss = np.zeros(arr.shape[0], dtype=np.int32)
    dicg = {'A': 0, 'G': 1, 'T': 2, 'C': 3}
    for i in range(arr.shape[0]):
        prev = "N"
        for j in range(arr.shape[1]):
            if arr[i, j] in "AGTC":
                if prev != arr[i, j]:
                    y_out[i, poss[i]] = dicg[arr[i, j]]
                    prev = arr[i, j]
                else:
                    y_out[i, poss[i]] = 4 + dicg[arr[i, j]]
                    prev = "N"

                poss[i] += 1
            elif arr[i, j] == 'N':
                pass
            else:
                raise ValueError("wtf")
    return y_out, poss


def decode(arr):
    arr = arr.astype(np.int32)
    y_out = np.zeros_like(arr, dtype=np.unicode)
    for i, a in enumerate("AGTCAGTCN"):
        y_out[arr == i] = a
    return y_out


def read_fast5(filename, truncate=5000):
    'This assumes we are in the right dir.'
    with h5py.File(os.path.join('pass', filename + '.fast5'), 'r') as h5:
        reads = h5['Analyses/EventDetection_000/Reads']
        events = np.array(reads[list(reads.keys())[0] + '/Events'])

        basecalled_events = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
        basecalled = np.array(basecalled_events.value[['mean', 'stdv', 'model_state', 'move', 'start',
            'length']])

        length = min(events.shape[0], truncate)
        if events.shape[0] < truncate:
            print("WARNING...less then truncate events", filename)
            return None

        # x[i] is feat values for event #i
        x = np.zeros((length, 3))
        # y[2*i] and y[2*i + 1] are bases for event #i
        y = np.array(['N'] * length)

        bcall_idx = 0
        write = 0
        for i, e in enumerate(events[:length]):
            write = max(write, i)
            if write >= length:
                print("WARN!!! Write", write, filename)
                return None
            if bcall_idx < basecalled.shape[0]:
                b = basecalled[bcall_idx]
                if b[0] == e[2] and b[1] == e[3]: # mean == mean and stdv == stdv
                    if bcall_idx == 0:
                        write = max(write - 5, 0)
                        y[write:write+5] = list(b[2].decode("ASCII")) # initial model state
                        write += 5
                    bcall_idx += 1
                    if b[3] == 1:
                        y[write] = chr(b[2][-1])
                        write += 1
                    if b[3] == 2:
                        y[write] = chr(b[2][-2])
                        if write + 1 >= length:
                            print("WARN!!! Write", write, filename)
                            return None
                        y[write + 1] = chr(b[2][-1])
                        write += 2

        x[:,0] = events['length'][:length]
        x[:,1] = events['mean'][:length]
        x[:,2] = events['stdv'][:length]
    return x, y


def gen_dummy_ds(size=100):
    files = list(map(lambda x:x[:-6], os.listdir('./pass')))
    random.shuffle(files)
    X, Y = [], []
    for file in files:
        if len(X) == size:
            break
        sol = read_fast5(file)
        if sol is not None:
            x, y = sol
            X.append(x)
            Y.append(y)
    np.savez_compressed(os.path.expanduser('~/dataset.npz'), X=X, Y=Y)

if __name__ == "__main__":
    X = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7]).reshape(1, 7, 1), dtype=tf.float32)
    kernel = tf.constant(np.array([100, 10, 1]).reshape(3, 1, 1), dtype=tf.float32)
    y = atrous_conv1d(X, kernel, 2, "SAME")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    gg = sess.run(y)
    print(gg, gg.shape)
