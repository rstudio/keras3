


#' Performs elementwise addition operation.
#'
#' @description
#' It takes as input a list of tensors, all of the same shape,
#' and returns a single tensor (also of the same shape).
#'
#' # Examples
#' ```{r}
#' input_shape <- c(1, 2, 3)
#' x1 <- op_ones(input_shape)
#' x2 <- op_ones(input_shape)
#' layer_add(x1, x2)
#' ```
#'
#' Usage in a Keras model:
#'
#' ```{r}
#' input1 <- layer_input(shape = c(16))
#' x1 <- input1 |> layer_dense(8, activation = 'relu')
#'
#' input2 <- layer_input(shape = c(32))
#' x2 <- input2 |> layer_dense(8, activation = 'relu')
#'
#' # equivalent to `added = layer_add([x1, x2))`
#' added <- layer_add(x1, x2)
#' output <- added |> layer_dense(4)
#'
#' model <- keras_model(inputs = c(input1, input2), outputs = output)
#' ```
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param inputs
#' layers to combine
#'
#' @inherit layer_dense return
#' @export
#' @family add merging layers
#' @family merging layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/merging_layers/add#add-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add>
#' @tether keras.layers.Add
layer_add <-
function (inputs, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- create_layer(keras$layers$Add, NULL, args)
    if (length(inputs))
        layer(inputs)
    else layer
}


#' Averages a list of inputs element-wise..
#'
#' @description
#' It takes as input a list of tensors, all of the same shape,
#' and returns a single tensor (also of the same shape).
#'
#' # Examples
#' ```{r}
#' input_shape <- c(1, 2, 3)
#' x1 <- op_ones(input_shape)
#' x2 <- op_zeros(input_shape)
#' layer_average(x1, x2)
#' ```
#'
#' Usage in a Keras model:
#'
#' ```{r}
#' input1 <- layer_input(shape = c(16))
#' x1 <- input1 |> layer_dense(8, activation = 'relu')
#'
#' input2 <- layer_input(shape = c(32))
#' x2 <- input2 |> layer_dense(8, activation = 'relu')
#'
#' added <- layer_average(x1, x2)
#' output <- added |> layer_dense(4)
#'
#' model <- keras_model(inputs = c(input1, input2), outputs = output)
#' ```
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param inputs
#' layers to combine
#'
#' @inherit layer_dense return
#' @export
#' @family average merging layers
#' @family merging layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/merging_layers/average#average-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Average>
#' @tether keras.layers.Average
layer_average <-
function (inputs, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- create_layer(keras$layers$Average, NULL, args)
    if (length(inputs))
        layer(inputs)
    else layer
}


#' Concatenates a list of inputs.
#'
#' @description
#' It takes as input a list of tensors, all of the same shape except
#' for the concatenation axis, and returns a single tensor that is the
#' concatenation of all inputs.
#'
#' # Examples
#' ```{r}
#' x <- op_arange(20) |> op_reshape(c(2, 2, 5))
#' y <- op_arange(21, 40) |> op_reshape(c(2, 2, 5))
#' layer_concatenate(x, y, axis = 2)
#' ```
#' Usage in a Keras model:
#'
#' ```{r}
#' x1 <- op_arange(10)     |> op_reshape(c(5, 2)) |> layer_dense(8)
#' x2 <- op_arange(11, 20) |> op_reshape(c(5, 2)) |> layer_dense(8)
#' y <- layer_concatenate(x1, x2)
#' ```
#'
#' @returns
#' A tensor, the concatenation of the inputs alongside axis `axis`.
#'
#' @param axis
#' Axis along which to concatenate.
#'
#' @param ...
#' Standard layer keyword arguments.
#'
#' @param inputs
#' layers to combine
#'
#' @export
#' @family concatenate merging layers
#' @family merging layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/merging_layers/concatenate#concatenate-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate>
#' @tether keras.layers.Concatenate
layer_concatenate <-
function (inputs, ..., axis = -1L)
{
    args <- capture_args(list(axis = as_axis, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- create_layer(keras$layers$Concatenate, NULL, args)
    if (length(inputs))
        layer(inputs)
    else layer
}


#' Computes element-wise dot product of two tensors.
#'
#' @description
#' It takes a list of inputs of size 2, and the axes
#' corresponding to each input along with the dot product
#' is to be performed.
#'
#' Let's say `x` and `y` are the two input tensors with shapes
#' `(2, 3, 5)` and `(2, 10, 3)`. The batch dimension should be
#' of same size for both the inputs, and `axes` should correspond
#' to the dimensions that have the same size in the corresponding
#' inputs. e.g. with `axes = c(1, 2)`, the dot product of `x`, and `y`
#' will result in a tensor with shape `(2, 5, 10)`
#'
#' # Examples
#'
#' ```{r}
#' x <- op_reshape(0:9,   c(1, 5, 2))
#' y <- op_reshape(10:19, c(1, 2, 5))
#' layer_dot(x, y, axes=c(2, 3))
#' ```
#'
#' Usage in a Keras model:
#'
#' ```{r}
#' x1 <- op_reshape(0:9, c(5, 2)) |> layer_dense(8)
#' x2 <- op_reshape(10:19, c(5, 2)) |> layer_dense(8)
#' shape(x1)
#' shape(x2)
#' y <- layer_dot(x1, x2, axes=2)
#' shape(y)
#' ```
#'
#' @returns
#' A tensor, the dot product of the samples from the inputs.
#'
#' @param axes
#' Integer or list of integers, axis or axes along which to
#' take the dot product. If a list, should be two integers
#' corresponding to the desired axis from the first input and the
#' desired axis from the second input, respectively. Note that the
#' size of the two selected axes must match.
#'
#' @param normalize
#' Whether to L2-normalize samples along the dot product axis
#' before taking the dot product. If set to `TRUE`, then
#' the output of the dot product is the cosine proximity
#' between the two samples.
#'
#' @param ...
#' Standard layer keyword arguments.
#'
#' @param inputs
#' layers to combine
#'
#' @export
#' @family dot merging layers
#' @family merging layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/merging_layers/dot#dot-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dot>
#' @tether keras.layers.Dot
layer_dot <-
function (inputs, ..., axes, normalize = FALSE)
{
    args <- capture_args(list(axes = as_axis, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- create_layer(keras$layers$Dot, NULL, args)
    if (length(inputs))
        layer(inputs)
    else layer
}


#' Computes element-wise maximum on a list of inputs.
#'
#' @description
#' It takes as input a list of tensors, all of the same shape,
#' and returns a single tensor (also of the same shape).
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 3, 4)
#' x1 <- random_uniform(input_shape)
#' x2 <- random_uniform(input_shape)
#' y <- layer_maximum(x1, x2)
#' ```
#'
#' Usage in a Keras model:
#'
#' ```{r}
#' input1 <- layer_input(shape = c(16))
#' x1 <- input1 |> layer_dense(8, activation = 'relu')
#' input2 <- layer_input(shape = c(32))
#' x2 <- input2 |> layer_dense(8, activation = 'relu')
#' # equivalent to `y <- layer_maximum(x1, x2)`
#' y <- layer_maximum(x1, x2)
#' out <- y |> layer_dense(4)
#' model <- keras_model(inputs = c(input1, input2), outputs = out)
#' ```
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param inputs
#' layers to combine
#'
#' @inherit layer_dense return
#' @export
#' @family maximum merging layers
#' @family merging layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/merging_layers/maximum#maximum-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Maximum>
#' @tether keras.layers.Maximum
layer_maximum <-
function (inputs, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- create_layer(keras$layers$Maximum, NULL, args)
    if (length(inputs))
        layer(inputs)
    else layer
}


#' Computes elementwise minimum on a list of inputs.
#'
#' @description
#' It takes as input a list of tensors, all of the same shape,
#' and returns a single tensor (also of the same shape).
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 3, 4)
#' x1 <- random_uniform(input_shape)
#' x2 <- random_uniform(input_shape)
#' y <- layer_minimum(x1, x2)
#' ```
#'
#' Usage in a Keras model:
#'
#' ```{r}
#' input1 <- layer_input(shape = c(16))
#' x1 <- input1 |> layer_dense(8, activation = 'relu')
#' input2 <- layer_input(shape = c(32))
#' x2 <- input2 |> layer_dense(8, activation = 'relu')
#' # equivalent to `y <- layer_minimum(x1, x2)`
#' y <- layer_minimum(x1, x2)
#' out <- y |> layer_dense(4)
#' model <- keras_model(inputs = c(input1, input2), outputs = out)
#' ```
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param inputs
#' layers to combine
#'
#' @inherit layer_dense return
#' @export
#' @family minimum merging layers
#' @family merging layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/merging_layers/minimum#minimum-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Minimum>
#' @tether keras.layers.Minimum
layer_minimum <-
function (inputs, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- create_layer(keras$layers$Minimum, NULL, args)
    if (length(inputs))
        layer(inputs)
    else layer
}


#' Performs elementwise multiplication.
#'
#' @description
#' It takes as input a list of tensors, all of the same shape,
#' and returns a single tensor (also of the same shape).
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 3, 4)
#' x1 <- random_uniform(input_shape)
#' x2 <- random_uniform(input_shape)
#' y <- layer_multiply(x1, x2)
#' ```
#'
#' Usage in a Keras model:
#'
#' ```{r}
#' input1 <- layer_input(shape = c(16))
#' x1 <- input1 |> layer_dense(8, activation = 'relu')
#' input2 <- layer_input(shape = c(32))
#' x2 <- input2 |> layer_dense(8, activation = 'relu')
#' # equivalent to `y <- layer_multiply(x1, x2)`
#' y <- layer_multiply(x1, x2)
#' out <- y |> layer_dense(4)
#' model <- keras_model(inputs = c(input1, input2), outputs = out)
#' ```
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param inputs
#' layers to combine
#'
#' @inherit layer_dense return
#' @export
#' @family multiply merging layers
#' @family merging layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/merging_layers/multiply#multiply-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Multiply>
#' @tether keras.layers.Multiply
layer_multiply <-
function (inputs, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- create_layer(keras$layers$Multiply, NULL, args)
    if (length(inputs))
        layer(inputs)
    else layer
}


#' Performs elementwise subtraction.
#'
#' @description
#' It takes as input a list of tensors of size 2 both of the
#' same shape, and returns a single tensor `(inputs[0] - inputs[1))`
#' of same shape.
#'
#' # Examples
#' ```{r}
#' input_shape <- c(2, 3, 4)
#' x1 <- random_uniform(input_shape)
#' x2 <- random_uniform(input_shape)
#' y <- layer_subtract(list(x1, x2))
#' ```
#'
#' Usage in a Keras model:
#'
#' ```{r}
#' input1 <- layer_input(shape = 16)
#' x1 <- layer_dense(input1, units = 8, activation = 'relu')
#' input2 <- layer_input(shape = 32)
#' x2 <- layer_dense(input2, units = 8, activation = 'relu')
#' subtracted <- layer_subtract(list(x1, x2))
#' out <- layer_dense(subtracted, units = 4)
#' model <- keras_model(inputs = list(input1, input2), outputs = out)
#' ```
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param inputs
#' layers to combine
#'
#' @inherit layer_dense return
#' @export
#' @family subtract merging layers
#' @family merging layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/merging_layers/subtract#subtract-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Subtract>
#'
#' @tether keras.layers.Subtract
layer_subtract <-
function (inputs, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = c("...", "inputs"))
    dots <- split_dots_named_unnamed(list(...))
    if (missing(inputs))
        inputs <- NULL
    else if (!is.null(inputs) && !is.list(inputs))
        inputs <- list(inputs)
    inputs <- c(inputs, dots$unnamed)
    args <- c(args, dots$named)
    layer <- create_layer(keras$layers$Subtract, NULL, args)
    if (length(inputs))
        layer(inputs)
    else layer
}
