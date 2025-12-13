# A regularizer that applies both L1 and L2 regularization penalties.

The L1 regularization penalty is computed as:
`loss = l1 * reduce_sum(abs(x))`

The L2 regularization penalty is computed as
`loss = l2 * reduce_sum(square(x))`

L1L2 may be passed to a layer as a string identifier:

    dense <- layer_dense(units = 3, kernel_regularizer = 'L1L2')

In this case, the default values used are `l1=0.01` and `l2=0.01`.

## Usage

``` r
regularizer_l1_l2(l1 = 0, l2 = 0)
```

## Arguments

- l1:

  float, L1 regularization factor.

- l2:

  float, L2 regularization factor.

## Value

A `Regularizer` instance that can be passed to layer constructors or
used as a standalone object.

## See also

- <https://keras.io/api/layers/regularizers#l1l2-class>

Other regularizers:  
[`regularizer_l1()`](https://keras3.posit.co/dev/reference/regularizer_l1.md)  
[`regularizer_l2()`](https://keras3.posit.co/dev/reference/regularizer_l2.md)  
[`regularizer_orthogonal()`](https://keras3.posit.co/dev/reference/regularizer_orthogonal.md)  
