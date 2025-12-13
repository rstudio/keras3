# Represents a backend-agnostic variable in Keras.

A `Variable` acts as a container for state. It holds a tensor value and
can be updated. With the JAX backend, variables are used to implement
"functionalization", the pattern of lifting stateful operations out of a
piece of computation to turn it into a stateless function.

## Usage

``` r
keras_variable(
  initializer,
  shape = NULL,
  dtype = NULL,
  trainable = TRUE,
  autocast = TRUE,
  aggregation = "none",
  synchronization = "auto",
  name = NULL,
  ...
)
```

## Arguments

- initializer:

  Initial value or callable for initialization. If a callable is used,
  it should take the arguments `shape` and `dtype`.

- shape:

  Optional. Tuple for the variable's shape. Required if `initializer` is
  a callable.

- dtype:

  Optional. Data type of the variable. Defaults to the global float
  dtype type (`"float32"` if never configured).

- trainable:

  Optional. Boolean indicating if variable is trainable. Defaults to
  `TRUE`.

- autocast:

  Optional. Boolean indicating whether the variable supports
  autocasting. If `TRUE`, the layer may first convert the variable to
  the compute data type when accessed. Defaults to `TRUE`.

- aggregation:

  Optional string, one of `NULL`, `"none"`, `"mean"`, `"sum"` or
  `"only_first_replica"` specifying how a distributed variable will be
  aggregated. This serves as a semantic annotation, to be taken into
  account by downstream backends or users. Defaults to `"none"`.

- synchronization:

  Optional string specifying how distributed values should be
  synchronized. Defaults to `"auto"`.

- name:

  Optional. A unique name for the variable. Automatically generated if
  not set.

- ...:

  Additional backend-specific keyword arguments forwarded to
  `keras$Variable()`.

## Attributes

- `shape`: The shape of the variable (tuple of integers).

- `ndim`: The number of dimensions of the variable (integer).

- `dtype`: The data type of the variable (string).

- `trainable`: Whether the variable is trainable (boolean).

- `autocast`: Whether the variable supports autocasting (boolean).

- `aggregation`: How a distributed variable will be aggregated (string).

- `synchronization`: Strategy for synchronizing the variable across
  devices (string).

- `value`: The current value of the variable (NumPy array or tensor).

- `name`: The name of the variable (string).

- `path`: The path of the variable within the Keras model or layer
  (string).

## Examples

**Initializing a `Variable` with a NumPy array:**

    initial_array <- array(1, c(3, 3))
    variable_from_array <- keras_variable(initializer = initial_array)

**Using a Keras initializer to create a `Variable`:**

    variable_from_initializer <- keras_variable(
      initializer = initializer_ones(),
      shape = c(3, 3),
      dtype = "float32"
    )

    new_value <- array(0, c(3, 3))
    variable_from_array$assign(new_value)

    ## tf.Tensor(
    ## [[0. 0. 0.]
    ##  [0. 0. 0.]
    ##  [0. 0. 0.]], shape=(3, 3), dtype=float64)

    # To modify a subset of values
    value <- variable_from_array$value
    value@r[1,] <- 99
    invisible(variable_from_array$assign(value))
    variable_from_array

    ## <Variable path=variable, shape=(3, 3), dtype=float64, value=[[99. 99. 99.]
    ##  [ 0.  0.  0.]
    ##  [ 0.  0.  0.]]>

**Marking a `Variable` as non-trainable:**

    non_trainable_variable <- keras_variable(
      initializer = array(1, c(3, 3)),
      dtype = "float32",
      trainable = FALSE
    )
