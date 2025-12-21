# Regularizer that encourages input vectors to be orthogonal to each other.

It can be applied to either the rows of a matrix (`mode="rows"`) or its
columns (`mode="columns"`). When applied to a `Dense` kernel of shape
`(input_dim, units)`, rows mode will seek to make the feature vectors
(i.e. the basis of the output space) orthogonal to each other.

## Usage

``` r
regularizer_orthogonal(factor = 0.01, mode = "rows")
```

## Arguments

- factor:

  Float. The regularization factor. The regularization penalty will be
  proportional to `factor` times the mean of the dot products between
  the L2-normalized rows (if `mode="rows"`, or columns if
  `mode="columns"`) of the inputs, excluding the product of each
  row/column with itself. Defaults to `0.01`.

- mode:

  String, one of `{"rows", "columns"}`. Defaults to `"rows"`. In rows
  mode, the regularization effect seeks to make the rows of the input
  orthogonal to each other. In columns mode, it seeks to make the
  columns of the input orthogonal to each other.

## Value

A `Regularizer` instance that can be passed to layer constructors or
used as a standalone object.

## Examples

    regularizer <- regularizer_orthogonal(factor=0.01)
    layer <- layer_dense(units=4, kernel_regularizer=regularizer)

## See also

- <https://keras.io/api/layers/regularizers#orthogonalregularizer-class>

Other regularizers:  
[`regularizer_l1()`](https://keras3.posit.co/reference/regularizer_l1.md)  
[`regularizer_l1_l2()`](https://keras3.posit.co/reference/regularizer_l1_l2.md)  
[`regularizer_l2()`](https://keras3.posit.co/reference/regularizer_l2.md)  
