# Constrains the weights incident to each hidden unit to have unit norm.

Constrains the weights incident to each hidden unit to have unit norm.

## Usage

``` r
constraint_unitnorm(axis = 1L)
```

## Arguments

- axis:

  integer, axis along which to calculate weight norms. For instance, in
  a `Dense` layer the weight matrix has shape `(input_dim, output_dim)`,
  set `axis` to `0` to constrain each weight vector of length
  `(input_dim,)`. In a `Conv2D` layer with
  `data_format = "channels_last"`, the weight tensor has shape
  `(rows, cols, input_depth, output_depth)`, set `axis` to `[0, 1, 2]`
  to constrain the weights of each filter tensor of size
  `(rows, cols, input_depth)`.

## Value

A `Constraint` instance, a callable that can be passed to layer
constructors or used directly by calling it with tensors.

## See also

- <https://keras.io/api/layers/constraints#unitnorm-class>

Other constraints:  
[`Constraint()`](https://keras3.posit.co/reference/Constraint.md)  
[`constraint_maxnorm()`](https://keras3.posit.co/reference/constraint_maxnorm.md)  
[`constraint_minmaxnorm()`](https://keras3.posit.co/reference/constraint_minmaxnorm.md)  
[`constraint_nonneg()`](https://keras3.posit.co/reference/constraint_nonneg.md)  
