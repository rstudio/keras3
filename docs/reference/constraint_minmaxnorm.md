# MinMaxNorm weight constraint.

Constrains the weights incident to each hidden unit to have the norm
between a lower bound and an upper bound.

## Usage

``` r
constraint_minmaxnorm(min_value = 0, max_value = 1, rate = 1, axis = 1L)
```

## Arguments

- min_value:

  the minimum norm for the incoming weights.

- max_value:

  the maximum norm for the incoming weights.

- rate:

  rate for enforcing the constraint: weights will be rescaled to yield
  op_clip?
  `(1 - rate) * norm + rate * op_clip(norm, min_value, max_value)`.
  Effectively, this means that rate = 1.0 stands for strict enforcement
  of the constraint, while rate\<1.0 means that weights will be rescaled
  at each step to slowly move towards a value inside the desired
  interval.

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

- <https://keras.io/api/layers/constraints#minmaxnorm-class>

Other constraints:  
[`Constraint()`](https://keras3.posit.co/reference/Constraint.md)  
[`constraint_maxnorm()`](https://keras3.posit.co/reference/constraint_maxnorm.md)  
[`constraint_nonneg()`](https://keras3.posit.co/reference/constraint_nonneg.md)  
[`constraint_unitnorm()`](https://keras3.posit.co/reference/constraint_unitnorm.md)  
