# Small crash course to RNN in tensorflow


Basic RNN have the following structure, where:
* $x_t$ input at time $t$
* $h_t$ hidden state at time $t$
* $\phi$ activation function, usually $\tanh$

$h_t$ is usually propagated to upper layers as output.

Usually beginning hidden state is initialized with zeros, that is $h_0 = \mathbf{0}$

Equations for vanilla RNN are as follows:

$$
\begin{align}
    h_t &=\phi(Wh_{t_1} + Ux_t + b) \\
\end{align}
$$

The problem is simply vanishing/exploding gradients since with back propagation through time (BPTT) and information morphing (We apply non-linear transformation to hidden state each time step and the network cannot rely on same information representation in later time step). Exploding/Vanishing gradients are due to constant multiplication with matrix $W$ and is further explained [here](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html#a-mathematically-sufficient-condition-for-vanishing-sensitivity)

Therefore, let's go furher to basic LSTM(Long short term memory) cell (I'm going to skip prototype LSTM cell which isn't working satisfactory due to unbounded state and other issues. See [here](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html#gluing-gates-together-to-derive-a-prototype-lstm))


Basic LSTM cell as implemented in tf is in class [tf.nn.rnn_cell.BasicLSTMCell](https://www.tensorflow.org/versions/r0.11/api_docs/python/rnn_cell.html#BasicLSTMCell). The internal architecture is as follows:

* $h_t$ - shadow state
* $c_t$ - cell
* $i_t$ - input gate
* $o_t$ - output gate
* $f_t$ - forget gate

$$
\begin{align}
    i_t &= \sigma(W_ih_{t-1} + U_i x_{t-1} + b_i) \\
    o_t &= \sigma(W_oh_{t-1} + U_o x_{t-1} + b_o) \\
    f_t &= \sigma(W_fh_{t-1} + U_f x_{t-1} + b_f) \\
    \\
    \tilde{c_t} &= \phi(W_ch_{t-1} + U_cx_t + b_c) \\
    c_t &= f_t \odot c_t + i_t \odot \tilde{c_t} \\
    \\
    h_t &= o_t \odot \phi(c_t)\\
\end{align}
$$

Other altrenative is GRU (Gated Recurrent Unit) which has similar architecture as LSTM, though input and forget gates are merged, that is linked together. As inplemented in [GRUCell](https://www.tensorflow.org/versions/r0.11/api_docs/python/rnn_cell.html#GRUCell):

* $z_t$ - update gate
* $r_t$ - reset gate
* $\tilde{h_t}$ - candidate write

$$
\begin{align}
    r_t &= \sigma(W_rh_{t-1} + U_r x_{t-1} + b_r) \\
    z_t &= \sigma(W_zh_{t-1} + U_z x_{t-1} + b_z) \\
    \tilde{h_t} &=  \phi(W_h (h_{t-1} \odot r_t) + U_h x_t + b_h)\\
    h_t &= z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h_t}
\end{align}
$$

# Tensorflow implementation details

For merging single RNN cell into multilayer RNN use [MultiRNNCell](https://www.tensorflow.org/versions/r0.11/api_docs/python/rnn_cell.html#MultiRNNCell).

For dropout (on input and output, never on the state itself!) use [DropoutWrapper](https://www.tensorflow.org/versions/r0.11/api_docs/python/rnn_cell.html#DropoutWrapper)

For constructing RNN from cells (either MultiRNNCell, or any other combination) use [tf.nn.dynamic_rnn](https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#dynamic_rnn). It constructs computation graph dynamically at runtime and it's faster than tf.nn.rnn. I want to bring extra attention on parameter sequence_length:

```
The parameter sequence_length is optional and is used to copy-through state and zero-out outputs when past a batch element's sequence length. So it's more for correctness than performance, unlike in rnn().
```

As far as I figured out, we supply already max_time truncated samples into and it performs training starting from same inital state...I'm not really sure how to go when we have long sequence which we need to use BPTT...I mean they are ....todo

[state_saving_rnn](https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#state_saving_rnn)... looks useful but I cannot really understand what's going on under the hood. After some googling I found [this](https://www.tensorflow.org/versions/master/api_docs/python/contrib.training.html#SequenceQueueingStateSaver) as example usage.. skip for now.

There's few more useful looking functions, have a look [here](https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#recurrent-neural-networks)

In [seq2seq](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py) there's useful looking stuff,  sequence_loss_by_example and potentially useful function.

# Sweet sweet code!!!

```python
#!/usr/bin/evn python3

import tensorflow as tf
import numpy as np

num_steps = 32
batch_size = 32
num_units = 100
num_layers = 3

X = tf.placeholder(tf.float32, [None, num_steps])
Y = tf.placeholder(tf.float32, [None, num_steps])

cell = tf.nn.rnn_cell.GRUCell(num_units)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)


init_state = cell.zero_state(batch_size, tf.float32)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [num_units, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

# A bit of reshaping magic
rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
y_reshaped = tf.reshape(y, [-1])

logits = tf.matmul(rnn_outputs, W) + b

predictions = tf.nn.softmax(logits)

total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
```

During traning, lets assume we have long sequences x, y, and we already subdivided them in smaller parts with size [batch_size, num_steps]. Then we train as:

```python
training_state = None
for x,y in zip(X, Y):
    feed_dict={g['x']: X, g['y']: Y}
    if training_state is not None:
        feed_dict[g['init_state']] = training_state
    _, sess.run([train_step, final_state], feed_dict)
```

# Other resources

* [Colah - Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - quick and dirty overview of basic RNN cells
* [Training recurrent neural network, Ilya Sutskever 2013 phd thesis](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
* [RNN chapter in Deep Learning Book](http://www.deeplearningbook.org/contents/rnn.html)
* [Official tf RNN tutorial](https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html)
* [RNN in tensorflow - a practical guide and undocumented features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/) -- tf documentation could be better with more examples, so I have to scrape everything from blogs mostly
* [R2RT - Written Memories: Understanding, Deriving and Extending the LSTM ](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html) -- excellent blog post with lot of math in it
* [R2RT - Recurrent Neural Networks in Tensorflow I](http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html) and [part 2](http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html) -- good resources with lot of practical tips, code and up to date constructs
* [Styles of Truncated Backpropagation
](http://r2rt.com/styles-of-truncated-backpropagation.html) -- explains styles of BPTT and which one tensorflow uses
