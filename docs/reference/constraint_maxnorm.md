# MaxNorm weight constraint.

Constrains the weights incident to each hidden unit to have a norm less
than or equal to a desired value.

## Usage

``` r
constraint_maxnorm(max_value = 2L, axis = 1L)
```

## Arguments

- max_value:

  the maximum norm value for the incoming weights.

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

- <https://keras.io/api/layers/constraints#maxnorm-class>

Other constraints:  
[`Constraint()`](https://keras3.posit.co/reference/Constraint.md)  
[`constraint_minmaxnorm()`](https://keras3.posit.co/reference/constraint_minmaxnorm.md)  
[`constraint_nonneg()`](https://keras3.posit.co/reference/constraint_nonneg.md)  
[`constraint_unitnorm()`](https://keras3.posit.co/reference/constraint_unitnorm.md)  
