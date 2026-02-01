# Tensor shape utility

This function can be used to get or create a tensor shape.

## Usage

``` r
shape(...)

# S3 method for class 'keras_shape'
format(x, ..., prefix = TRUE)

# S3 method for class 'keras_shape'
print(x, ...)

# S3 method for class 'keras_shape'
x[...]

# S3 method for class 'keras_shape'
as.integer(x, ...)

# S3 method for class 'keras_shape'
Summary(..., na.rm = FALSE)

# S3 method for class 'keras_shape'
as.list(x, ...)

# S3 method for class 'keras_shape'
x == y

# S3 method for class 'keras_shape'
x != y
```

## Arguments

- ...:

  A shape specification. Numerics, `NULL` and tensors are valid. `NULL`,
  `NA`, and `-1L` can be used to specify an unspecified dim size.
  Tensors are dispatched to
  [`op_shape()`](https://keras3.posit.co/reference/op_shape.md) to
  extract the tensor shape. Values wrapped in
  [`I()`](https://rdrr.io/r/base/AsIs.html) are used asis (see
  examples). All other objects are coerced via
  [`as.integer()`](https://rdrr.io/r/base/integer.html).

- x, y:

  A `keras_shape` object.

- prefix:

  Whether to format the shape object with a prefix. Defaults to
  `"shape"`.

- na.rm:

  passed on to Summary group generics like
  [`prod()`](https://rdrr.io/r/base/prod.html). Unknown axes are treated
  as `NA`.

## Value

A list with a `"keras_shape"` class attribute. Each element of the list
will be either a) `NULL`, b) an R integer or c) a scalar integer tensor
(e.g., when supplied a TF tensor with an unspecified dimension in a
function being traced).

## Examples

    shape(1, 2, 3)

    ## shape(1, 2, 3)

3 ways to specify an unknown dimension

    shape(NA,   2, 3)
    shape(NULL, 2, 3)
    shape(-1,   2, 3)

    ## shape(NA, 2, 3)
    ## shape(NA, 2, 3)
    ## shape(NA, 2, 3)

Most functions that take a 'shape' argument also coerce with `shape()`

    layer_input(c(1, 2, 3))
    layer_input(shape(1, 2, 3))

    ## <KerasTensor shape=(None, 1, 2, 3), dtype=float32, sparse=False, ragged=False, name=keras_tensor>
    ## <KerasTensor shape=(None, 1, 2, 3), dtype=float32, sparse=False, ragged=False, name=keras_tensor_1>

You can also use `shape()` to get the shape of a tensor (excepting
scalar integer tensors).

    symbolic_tensor <- layer_input(shape(1, 2, 3))
    shape(symbolic_tensor)

    ## shape(NA, 1, 2, 3)

    eager_tensor <- op_ones(c(1,2,3))
    shape(eager_tensor)

    ## shape(1, 2, 3)

    op_shape(eager_tensor)

    ## shape(1, 2, 3)

Combine or expand shapes

    shape(symbolic_tensor, 4)

    ## shape(NA, 1, 2, 3, 4)

    shape(5, symbolic_tensor, 4)

    ## shape(5, NA, 1, 2, 3, 4)

Scalar integer tensors are treated as axis values. These are most
commonly encountered when tracing a function in graph mode, where an
axis size might be unknown.

    tfn <- tensorflow::tf_function(function(x) {
      print(op_shape(x))
      x
    },
    input_signature = list(tensorflow::tf$TensorSpec(shape(1, NA, 3))))
    invisible(tfn(op_ones(shape(1, 2, 3))))

    ## shape(1, Tensor("strided_slice:0", shape=(), dtype=int32), 3)

A useful pattern is to unpack the `shape()` with `%<-%`, like this:

    c(batch_size, seq_len, channels) %<-% shape(x)

    # `%<-%` also has support for skipping values
    # during unpacking with `.` and `...`. For example,
    # To retrieve just the first and/or last dim:
    c(batch_size, ...) %<-% shape(x)
    c(batch_size, ., .) %<-% shape(x)
    c(..., channels) %<-% shape(x)
    c(batch_size, ..., channels) %<-% shape(x)
    c(batch_size, ., channels) %<-% shape(x)

    echo_print <- function(x) {
      message("> ", deparse(substitute(x)));
      if(!is.null(x)) print(x)
    }
    tfn <- tensorflow::tf_function(function(x) {
      c(axis1, axis2, axis3) %<-% shape(x)
      echo_print(str(list(axis1 = axis1, axis2 = axis2, axis3 = axis3)))

      echo_print(shape(axis1))               # use axis1 tensor as axis value
      echo_print(shape(axis1, axis2, axis3)) # use axis1 tensor as axis value

      # use shape() to compose a new shape, e.g., in multihead attention
      n_heads <- 4
      echo_print(shape(axis1, axis2, n_heads, axis3/n_heads))

      x
    },
    input_signature = list(tensorflow::tf$TensorSpec(shape(NA, 4, 16))))
    invisible(tfn(op_ones(shape(2, 4, 16))))

    ## > str(list(axis1 = axis1, axis2 = axis2, axis3 = axis3))

    ## List of 3
    ##  $ axis1:<tf.Tensor 'strided_slice:0' shape=() dtype=int32>
    ##  $ axis2: int 4
    ##  $ axis3: int 16

    ## > shape(axis1)

    ## shape(Tensor("strided_slice:0", shape=(), dtype=int32))

    ## > shape(axis1, axis2, axis3)

    ## shape(Tensor("strided_slice:0", shape=(), dtype=int32), 4, 16)

    ## > shape(axis1, axis2, n_heads, axis3/n_heads)

    ## shape(Tensor("strided_slice:0", shape=(), dtype=int32), 4, 4, 4)

If you want to resolve the shape of a tensor that can potentially be a
scalar integer, you can wrap the tensor in
[`I()`](https://rdrr.io/r/base/AsIs.html), or use
[`op_shape()`](https://keras3.posit.co/reference/op_shape.md).

    (x <- op_convert_to_tensor(2L))

    ## tf.Tensor(2, shape=(), dtype=int32)

    # by default, shape() treats scalar integer tensors as axis values
    shape(x)

    ## shape(tf.Tensor(2, shape=(), dtype=int32))

    # to access the shape of a scalar integer,
    # call `op_shape()`, or protect with `I()`
    op_shape(x)

    ## shape()

    shape(I(x))

    ## shape()

## See also

[`op_shape()`](https://keras3.posit.co/reference/op_shape.md)
