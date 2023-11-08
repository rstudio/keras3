keras.regularizers.L2
__signature__
(l2=0.01)
__doc__
A regularizer that applies a L2 regularization penalty.

The L2 regularization penalty is computed as:
`loss = l2 * reduce_sum(square(x))`

L2 may be passed to a layer as a string identifier:

>>> dense = Dense(3, kernel_regularizer='l2')

In this case, the default value used is `l2=0.01`.

Arguments:
    l2: float, L2 regularization factor.
