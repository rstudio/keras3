MaxNorm weight constraint.

@description
Constrains the weights incident to each hidden unit
to have a norm less than or equal to a desired value.

Also available via the shortcut function `keras.constraints.max_norm`.

@param max_value
the maximum norm value for the incoming weights.

@param axis
integer, axis along which to calculate weight norms.
For instance, in a `Dense` layer the weight matrix
has shape `(input_dim, output_dim)`,
set `axis` to `0` to constrain each weight vector
of length `(input_dim,)`.
In a `Conv2D` layer with `data_format = "channels_last"`,
the weight tensor has shape
`(rows, cols, input_depth, output_depth)`,
set `axis` to `[0, 1, 2]`
to constrain the weights of each filter tensor of size
`(rows, cols, input_depth)`.

@export
@family constraints
@seealso
+ <https:/keras.io/api/layers/constraints#maxnorm-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/constraints/MaxNorm>
