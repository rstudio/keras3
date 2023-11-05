A regularizer that applies both L1 and L2 regularization penalties.

The L1 regularization penalty is computed as:
`loss = l1 * reduce_sum(abs(x))`

The L2 regularization penalty is computed as
`loss = l2 * reduce_sum(square(x))`

L1L2 may be passed to a layer as a string identifier:

>>> dense = Dense(3, kernel_regularizer='l1_l2')

In this case, the default values used are `l1=0.01` and `l2=0.01`.

Arguments:
    l1: float, L1 regularization factor.
    l2: float, L2 regularization factor.
