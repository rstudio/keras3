# A regularizer that applies a L1 regularization penalty.

The L1 regularization penalty is computed as:
`loss = l1 * reduce_sum(abs(x))`

L1 may be passed to a layer as a string identifier:

    dense <- layer_dense(units = 3, kernel_regularizer = 'l1')

In this case, the default value used is `l1=0.01`.

## Usage

``` r
regularizer_l1(l1 = 0.01)
```

## Arguments

- l1:

  float, L1 regularization factor.

## Value

A `Regularizer` instance that can be passed to layer constructors or
used as a standalone object.

## See also

- <https://keras.io/api/layers/regularizers#l1-class>

Other regularizers:  
[`regularizer_l1_l2()`](https://keras3.posit.co/reference/regularizer_l1_l2.md)  
[`regularizer_l2()`](https://keras3.posit.co/reference/regularizer_l2.md)  
[`regularizer_orthogonal()`](https://keras3.posit.co/reference/regularizer_orthogonal.md)  
