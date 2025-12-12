# A regularizer that applies a L2 regularization penalty.

The L2 regularization penalty is computed as:
`loss = l2 * reduce_sum(square(x))`

L2 may be passed to a layer as a string identifier:

    dense <- layer_dense(units = 3, kernel_regularizer='l2')

In this case, the default value used is `l2=0.01`.

## Usage

``` r
regularizer_l2(l2 = 0.01)
```

## Arguments

- l2:

  float, L2 regularization factor.

## Value

A `Regularizer` instance that can be passed to layer constructors or
used as a standalone object.

## See also

- <https://keras.io/api/layers/regularizers#l2-class>

Other regularizers:  
[`regularizer_l1()`](https://keras3.posit.co/dev/reference/regularizer_l1.md)  
[`regularizer_l1_l2()`](https://keras3.posit.co/dev/reference/regularizer_l1_l2.md)  
[`regularizer_orthogonal()`](https://keras3.posit.co/dev/reference/regularizer_orthogonal.md)  
