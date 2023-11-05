A regularizer that applies a L1 regularization penalty.

The L1 regularization penalty is computed as:
`loss = l1 * reduce_sum(abs(x))`

L1 may be passed to a layer as a string identifier:

>>> dense = Dense(3, kernel_regularizer='l1')

In this case, the default value used is `l1=0.01`.

Arguments:
    l1: float, L1 regularization factor.
