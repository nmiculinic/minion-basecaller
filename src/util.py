from edlib import Edlib
import h5py
import numpy as np
import tensorflow as tf


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


def next_num(prev, symbol):
    val = {
        'A': 0,
        'G': 1,
        'T': 2,
        'C': 3,
    }[symbol]
    if symbol == prev:
        return 'N', val + 4
    else:
        return symbol, val


def read_fast5(filename, block_size, num_blocks, warn_if_short=False):
    'Read fast5 file.'
    with h5py.File(filename, 'r') as h5:
        reads = h5['Analyses/EventDetection_000/Reads']
        events = np.array(reads[list(reads.keys())[0] + '/Events'])

        basecalled_events = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
        basecalled = np.array(basecalled_events.value[['mean', 'stdv', 'model_state', 'move', 'start', 'length']])

        length = block_size * num_blocks
        if warn_if_short and events.shape[0] < length:
            print("WARNING...less then truncate events", filename)

        # x[i] is feat values for event #i
        x = np.zeros([length, 3], dtype=np.float32)
        x_len = min(length, events.shape[0])

        # y[2*i] and y[2*i + 1] are bases for event #i
        y = np.zeros([length], dtype=np.uint8)
        y_len = np.zeros([num_blocks], dtype=np.int32)

        bcall_idx = 0
        prev, curr_sec = "N", 0
        for i, e in enumerate(events[:length]):
            if i // block_size > curr_sec:
                prev, curr_sec = "N", i // block_size

            if bcall_idx < basecalled.shape[0]:
                b = basecalled[bcall_idx]
                if b[0] == e[2] and b[1] == e[3]:  # mean == mean and stdv == stdv
                    add_chr = []
                    if bcall_idx == 0:
                        add_chr.extend(list(b[2].decode("ASCII")))  # initial model state
                    bcall_idx += 1
                    if b[3] == 1:
                        add_chr.append(chr(b[2][-1]))
                    if b[3] == 2:
                        add_chr.append(chr(b[2][-2]))
                        add_chr.append(chr(b[2][-1]))
                    # print(add_chr)
                    for c in add_chr:
                        prev, sym = next_num(prev, c)
                        y[curr_sec * block_size + y_len[curr_sec]] = sym
                        y_len[curr_sec] += 1
                    if y_len[curr_sec] > block_size:
                        print("Too many events in block!")
                        return None

        x[:events.shape[0], 0] = events['length'][:length]
        x[:events.shape[0], 1] = events['mean'][:length]
        x[:events.shape[0], 2] = events['stdv'][:length]

        # Normalizing data to 0 mean, 1 std
        means = np.array([9.2421999, 104.08872223, 2.02581143], dtype=np.float32)
        stds = np.array([4.38210583, 16.13312531, 1.82191491], dtype=np.float32)

        x -= means
        x /= stds

    return x, x_len, y, y_len


def read_fast5_raw(filename, block_size_x, block_size_y, num_blocks, warn_if_short=False):
    'This assumes we are in the right dir.'
    with h5py.File(filename, 'r', driver='core') as h5:
        reads = h5['Analyses/EventDetection_000/Reads']
        target_read = list(reads.keys())[0]
        events = np.array(reads[target_read + '/Events'])
        start_time = events['start'][0]
        start_pad = int(start_time - h5['Raw/Reads/' + target_read].attrs['start_time'])

        basecalled_events = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
        basecalled = np.array(basecalled_events.value[['mean', 'stdv', 'model_state', 'move', 'start', 'length']])

        signal = h5['Raw/Reads/' + target_read]['Signal']
        signal_len = h5['Raw/Reads/' + target_read].attrs['duration'] - start_pad

        x = np.zeros([block_size_x * num_blocks, 1], dtype=np.float32)
        x_len = min(signal_len, block_size_x * num_blocks)
        x[:x_len, 0] = signal[start_pad:start_pad + x_len]

        if len(signal) != start_pad + np.sum(events['length']):
            print(filename + " failed sanity check")  # Sanity check
            assert (len(signal) == start_pad + np.sum(events['length']))  # Sanity check
            assert (False)

        y = np.zeros([block_size_y * num_blocks], dtype=np.uint8)
        y_len = np.zeros([num_blocks], dtype=np.int32)

        bcall_idx = 0
        prev, curr_sec = "N", 0
        for e in events:
            if (e['start'] - start_time) // block_size_x > curr_sec:
                prev, curr_sec = "N", (e['start'] - start_time) // block_size_x
            if curr_sec >= num_blocks:
                break

            if bcall_idx < basecalled.shape[0]:
                b = basecalled[bcall_idx]
                if b[0] == e[2] and b[1] == e[3]:  # mean == mean and stdv == stdv
                    add_chr = []
                    if bcall_idx == 0:
                        add_chr.extend(list(b[2].decode("ASCII")))  # initial model state
                    bcall_idx += 1
                    if b[3] == 1:
                        add_chr.append(chr(b[2][-1]))
                    if b[3] == 2:
                        add_chr.append(chr(b[2][-2]))
                        add_chr.append(chr(b[2][-1]))
                    for c in add_chr:
                        prev, sym = next_num(prev, c)
                        y[curr_sec * block_size_y + y_len[curr_sec]] = sym
                        y_len[curr_sec] += 1
                    if y_len[curr_sec] > block_size_y:
                        print("Too many events in block!")
                        return None
    x -= 646.11133
    x /= 75.673653
    return x, x_len, y, y_len


def decode(arr):
    arr = arr.astype(np.int32)
    y_out = np.zeros_like(arr, dtype=np.unicode)
    for i, a in enumerate("AGTCAGTCN"):
        y_out[arr == i] = a
    return y_out


def decode_sparse(arr, pad=None):
    indices = arr.indices
    if pad is None:
        pad = np.max(indices[:, 1]) + 1
    values = decode(arr.values)
    shape = arr.shape

    tmp = np.array([[' '] * pad] * shape[0])
    for ind, val in zip(indices, values):
        r, c = ind
        tmp[r, c] = val
    sol = np.array([' ' * pad] * shape[0])
    for row in range(shape[0]):
        sol[row] = "".join([c for c in tmp[row]])
    return sol


def decode_example(Y, Y_len, num_blocks, block_size_y, pad=None):
    gg = []
    for blk in range(num_blocks):
        gg.append("".join([str(x) for x in decode(Y[blk * block_size_y:blk * block_size_y + Y_len[blk]].ravel())]))

    if pad is None:
        pad = np.max(list(map(len, gg)))

    return list(map(lambda x: x.ljust(pad, ' '), gg))


def dense2d_to_sparse(dense_input, length, name=None, dtype=None):
    with tf.name_scope(name, "dense2d_to_sparse"):
        num_batches = dense_input.get_shape()[0]
        print(dense_input.get_shape())
        print(length.get_shape())

        indices = [tf.pack([tf.fill([length[x]], x), tf.range(length[x])], axis=1) for x in range(num_batches)]
        indices = tf.concat(0, indices)
        indices = tf.to_int64(indices)

        values = [tf.squeeze(tf.slice(dense_input, [x, 0], [1, length[x]]), squeeze_dims=[0]) for x in
                  range(num_batches)]
        values = tf.concat(0, values)

        if dtype is not None:
            values = tf.cast(values, dtype)

        return tf.SparseTensor(indices, values, tf.to_int64(tf.shape(dense_input)))


""" Working with aligned refs """


def get_basecalled_sequence(basecalled_events):
    basecalled = np.array(basecalled_events.value[['mean', 'stdv', 'model_state', 'move', 'start', 'length']])
    seq = []
    for bcall_idx, b in enumerate(basecalled):
        add_chr = []
        if bcall_idx == 0:
            add_chr.extend(list(b[2].decode("ASCII")))  # initial model state
        elif b[3] == 1:
            add_chr.append(chr(b[2][-1]))
        elif b[3] == 2:
            add_chr.append(chr(b[2][-2]))
            add_chr.append(chr(b[2][-1]))

        seq.extend(add_chr)
    return ''.join(seq)


def init_aligment_end_struct(ref_seq, called_seq, aligment_seq):
    # called_seq[i] alignes up to aligned_upto[i] index in ref_seq

    aligned_upto = []
    ref_idx = 0

    for c in aligment_seq:
        if c == Edlib.EDLIB_EDOP_MATCH or c == Edlib.EDLIB_EDOP_MISMATCH:
            aligned_upto.append(ref_idx)
            ref_idx += 1

        elif c == Edlib.EDLIB_EDOP_INSERT:
            aligned_upto.append(max(0, ref_idx - 1))

        elif c == Edlib.EDLIB_EDOP_DELETE:
            ref_idx += 1

    aligned_upto.append(aligned_upto[-1])
    return aligned_upto


def _transform_multiples(seq):
    prev = 'N'
    transformed = []
    for c in seq:
        prev, sym = next_num(prev, chr(c))
        transformed.append(sym)
    return transformed


def extract_blocks(ref_seq, called_seq, events_len, block_size, num_blocks):
    aligner = Edlib()
    result = aligner.align(called_seq, ref_seq)
    aligment_seq = result.alignment
    aligment_end = init_aligment_end_struct(ref_seq, called_seq, aligment_seq)
    ref_seq = np.fromstring(ref_seq, np.int8)

    y = np.zeros([num_blocks * block_size], dtype=np.uint8)
    y_len = np.zeros([num_blocks], dtype=np.int32)

    sum_len = 0
    for i, curr_len in enumerate(events_len):
        if curr_len == 0:
            continue

        ref_start = aligment_end[sum_len]
        ref_end = aligment_end[sum_len + curr_len - 1] + 1
        ref_block = slice(ref_start, ref_end)

        n_bases = ref_end - ref_start
        y_block = slice(i * block_size, i * block_size + n_bases)

        y[y_block] = _transform_multiples(ref_seq[ref_block])

        assert np.all(
            y[i * block_size: i * block_size + n_bases - 1] !=
            y[1 + i * block_size: i * block_size + n_bases]
        )

        y_len[i] = n_bases
        sum_len += curr_len

    return y, y_len


def read_fast5_ref(fast5_path, ref_path, block_size, num_blocks, warn_if_short=False):
    'Read fast5 file.'

    with h5py.File(fast5_path, 'r') as h5, open(ref_path, 'r') as ref_file:
        reads = h5['Analyses/EventDetection_000/Reads']
        events = np.array(reads[list(reads.keys())[0] + '/Events'])

        basecalled_events = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
        basecalled = np.array(basecalled_events.value[['mean', 'stdv', 'model_state', 'move', 'start', 'length']])

        length = block_size * num_blocks
        if warn_if_short and events.shape[0] < length:
            print("WARNING...less then truncate events", fast5_path)

        x = np.zeros([length, 3], dtype=np.float32)
        x_len = min(length, events.shape[0])

        y = np.zeros([length], dtype=np.uint8)
        events_len = np.zeros([num_blocks], dtype=np.int32)

        bcall_idx = 0
        prev, curr_sec = "N", 0

        for i, e in enumerate(events[:length]):
            if i // block_size > curr_sec:
                prev, curr_sec = "N", i // block_size

            if bcall_idx < basecalled.shape[0]:
                b = basecalled[bcall_idx]

                if b[0] == e[2] and b[1] == e[3]:  # mean == mean and stdv == stdv
                    added_bases = 0
                    if bcall_idx == 0:
                        added_bases = 5
                        assert len(list(b[2].decode("ASCII"))) == 5

                    bcall_idx += 1
                    assert 0 <= b[3] <= 2
                    added_bases += b[3]
                    events_len[curr_sec] += added_bases

        ref_seq = ref_file.readlines()[3].strip()
        called_seq = get_basecalled_sequence(basecalled_events)
        y, y_len = extract_blocks(ref_seq, called_seq, events_len, block_size, num_blocks)

        if any(y_len > block_size):
            print("Too many events in block!")
            return None

        x[:events.shape[0], 0] = events['length'][:length]
        x[:events.shape[0], 1] = events['mean'][:length]
        x[:events.shape[0], 2] = events['stdv'][:length]

        # Normalizing data to 0 mean, 1 std
        means = np.array([9.2421999, 104.08872223, 2.02581143], dtype=np.float32)
        stds = np.array([4.38210583, 16.13312531, 1.82191491], dtype=np.float32)

        x -= means
        x /= stds

    return x, x_len, y, y_len


def read_fast5_raw_ref(fast5_path, ref_path, block_size_x, block_size_y, num_blocks, warn_if_short=False):
    with h5py.File(fast5_path, 'r') as h5, open(ref_path, 'r') as ref_file:
        reads = h5['Analyses/EventDetection_000/Reads']
        target_read = list(reads.keys())[0]
        events = np.array(reads[target_read + '/Events'])
        start_time = events['start'][0]
        start_pad = int(start_time - h5['Raw/Reads/' + target_read].attrs['start_time'])

        basecalled_events = h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
        basecalled = np.array(basecalled_events.value[['mean', 'stdv', 'model_state', 'move', 'start', 'length']])

        signal = h5['Raw/Reads/' + target_read]['Signal']
        signal_len = h5['Raw/Reads/' + target_read].attrs['duration'] - start_pad

        x = np.zeros([block_size_x * num_blocks, 1], dtype=np.float32)
        x_len = min(signal_len, block_size_x * num_blocks)
        x[:x_len, 0] = signal[start_pad:start_pad + x_len]

        if len(signal) != start_pad + np.sum(events['length']):
            print(fast5_path + " failed sanity check")  # Sanity check
            assert (len(signal) == start_pad + np.sum(events['length']))  # Sanity check
            assert (False)

        events_len = np.zeros([num_blocks], dtype=np.int32)

        bcall_idx = 0
        prev, curr_sec = "N", 0
        for e in events:
            if (e['start'] - start_time) // block_size_x > curr_sec:
                prev, curr_sec = "N", (e['start'] - start_time) // block_size_x
            if curr_sec >= num_blocks:
                break

            if bcall_idx < basecalled.shape[0]:
                b = basecalled[bcall_idx]

                if b[0] == e[2] and b[1] == e[3]:  # mean == mean and stdv == stdv
                    added_bases = 0

                    if bcall_idx == 0:
                        added_bases = 5
                        assert len(list(b[2].decode("ASCII"))) == 5

                    bcall_idx += 1
                    assert 0 <= b[3] <= 2
                    added_bases += b[3]
                    events_len[curr_sec] += added_bases

        ref_seq = ref_file.readlines()[3].strip()
        called_seq = get_basecalled_sequence(basecalled_events)
        y, y_len = extract_blocks(ref_seq, called_seq, events_len, block_size_y, num_blocks)

        if any(y_len > block_size_y):
            print("Too many events in block!")
            return None

    x -= 646.11133
    x /= 75.673653
    return x, x_len, y, y_len


if __name__ == "__main__":
    X = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7]).reshape(1, 7, 1), dtype=tf.float32)
    kernel = tf.constant(np.array([100, 10, 1]).reshape(3, 1, 1), dtype=tf.float32)
    y = atrous_conv1d(X, kernel, 2, "SAME")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    gg = sess.run(y)
    print(gg, gg.shape)
