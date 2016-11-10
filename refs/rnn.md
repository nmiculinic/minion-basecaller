# Small crash course to RNN in tensorflow


Basic RNN have the following structure, where:
* $x_t$ input at time $t$
* $y_t$ output at time $t$
* $h_t$ hidden state at time $t$

Usually beginning hidden state is initialized with zeros, that is $h_0 = \mathbf{0}$

Equations for vanilla RNN are as follows:

$$
h_t =\phi(Wh_{t_1} + Ux_t + b) \\
y_t = h_t
$$

The problem is simply vanishing/exploding gradients since with back propagation through time (BPTT) and information morphing (We apply non-linear transformation to hidden state each time step and the network cannot rely on same information representation in later time step). Exploding/Vanishing gradients are due to constant multiplication with matrix $W$ and is further explained [here](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html#a-mathematically-sufficient-condition-for-vanishing-sensitivity)

Therefore, let's go furher to basic LSTM cell (I'm going to skip prototype LSTM cell which isn't working satisfactory due to unbounded state and other issues. See [here](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html#gluing-gates-together-to-derive-a-prototype-lstm))


Basic LSTM cell 


# Other resources

* [Training recurrent neural network, Ilya Sutskever 2013 phd thesis](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
* [RNN chapter in Deep Learning Book](http://www.deeplearningbook.org/contents/rnn.html)
* [Official tf RNN tutorial](https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html)
* [RNN in tensorflow - a practical guide and undocumented features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/) -- tf documentation could be better with more examples, so I have to scrape everything from blogs mostly
* [R2RT - Written Memories: Understanding, Deriving and Extending the LSTM ](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html) -- excellent blog post with lot of math in it
* [R2RT - Recurrent Neural Networks in Tensorflow I](http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html) and [part 2](http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html) -- good resources with lot of practical tips, code and up to date constructs
* [Styles of Truncated Backpropagation
](http://r2rt.com/styles-of-truncated-backpropagation.html) -- explains styles of BPTT and which one tensorflow uses
