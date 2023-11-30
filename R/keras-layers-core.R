


#' Just your regular densely-connected NN layer.
#'
#' @description
#' `Dense` implements the operation:
#' `output = activation(dot(input, kernel) + bias)`
#' where `activation` is the element-wise activation function
#' passed as the `activation` argument, `kernel` is a weights matrix
#' created by the layer, and `bias` is a bias vector created by the layer
#' (only applicable if `use_bias` is `TRUE`).
#'
#' # Note
#' If the input to the layer has a rank greater than 2, `Dense`
#' computes the dot product between the `inputs` and the `kernel` along the
#' last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
#' For example, if input has dimensions `(batch_size, d0, d1)`, then we create
#' a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
#' of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
#' `batch_size * d0` such sub-tensors). The output in this case will have
#' shape `(batch_size, d0, units)`.
#'
#' # Input Shape
#' N-D tensor with shape: `(batch_size, ..., input_dim)`.
#' The most common situation would be
#' a 2D input with shape `(batch_size, input_dim)`.
#'
#' # Output Shape
#' N-D tensor with shape: `(batch_size, ..., units)`.
#' For instance, for a 2D input with shape `(batch_size, input_dim)`,
#' the output would have shape `(batch_size, units)`.
#'
#' @param units
#' Positive integer, dimensionality of the output space.
#'
#' @param activation
#' Activation function to use.
#' If you don't specify anything, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#'
#' @param use_bias
#' Boolean, whether the layer uses a bias vector.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix.
#'
#' @param bias_initializer
#' Initializer for the bias vector.
#'
#' @param kernel_regularizer
#' Regularizer function applied to
#' the `kernel` weights matrix.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#'
#' @param activity_regularizer
#' Regularizer function applied to
#' the output of the layer (its "activation").
#'
#' @param kernel_constraint
#' Constraint function applied to
#' the `kernel` weights matrix.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family core layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/core_layers/dense#dense-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>
#' @tether keras.layers.Dense
layer_dense <-
function (object, units, activation = NULL, use_bias = TRUE,
    kernel_initializer = "glorot_uniform", bias_initializer = "zeros",
    kernel_regularizer = NULL, bias_regularizer = NULL, activity_regularizer = NULL,
    kernel_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args2(list(units = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Dense, object, args)
}


#' A layer that uses `einsum` as the backing computation.
#'
#' @description
#' This layer can perform einsum calculations of arbitrary dimensionality.
#'
#' # Examples
#' **Biased dense layer with einsums**
#'
#' This example shows how to instantiate a standard Keras dense layer using
#' einsum operations. This example is equivalent to
#' `layer_Dense(64, use_bias=TRUE)`.
#'
#' ```{r}
#' input <- layer_input(shape = c(32))
#' output <- input |>
#'   layer_einsum_dense("ab,bc->ac",
#'                      output_shape = 64,
#'                      bias_axes = "c")
#' output # shape(NA, 64)
#' ```
#'
#' **Applying a dense layer to a sequence**
#'
#' This example shows how to instantiate a layer that applies the same dense
#' operation to every element in a sequence. Here, the `output_shape` has two
#' values (since there are two non-batch dimensions in the output); the first
#' dimension in the `output_shape` is `NA`, because the sequence dimension
#' `b` has an unknown shape.
#'
#' ```{r}
#' input <- layer_input(shape = c(32, 128))
#' output <- input |>
#'   layer_einsum_dense("abc,cd->abd",
#'                      output_shape = c(NA, 64),
#'                      bias_axes = "d")
#' output  # shape(NA, 32, 64)
#' ```
#'
#' **Applying a dense layer to a sequence using ellipses**
#'
#' This example shows how to instantiate a layer that applies the same dense
#' operation to every element in a sequence, but uses the ellipsis notation
#' instead of specifying the batch and sequence dimensions.
#'
#' Because we are using ellipsis notation and have specified only one axis, the
#' `output_shape` arg is a single value. When instantiated in this way, the
#' layer can handle any number of sequence dimensions - including the case
#' where no sequence dimension exists.
#'
#' ```{r}
#' input <- layer_input(shape = c(32, 128))
#' output <- input |>
#'   layer_einsum_dense("...x,xy->...y",
#'                      output_shape = 64,
#'                      bias_axes = "y")
#'
#' output  # shape(NA, 32, 64)
#' ```
#'
#' @param equation
#' An equation describing the einsum to perform.
#' This equation must be a valid einsum string of the form
#' `ab,bc->ac`, `...ab,bc->...ac`, or
#' `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
#' axis expression sequence.
#'
#' @param output_shape
#' The expected shape of the output tensor
#' (excluding the batch dimension and any dimensions
#' represented by ellipses). You can specify `NA` or `NULL` for any dimension
#' that is unknown or can be inferred from the input shape.
#'
#' @param activation
#' Activation function to use. If you don't specify anything,
#' no activation is applied
#' (that is, a "linear" activation: `a(x) = x`).
#'
#' @param bias_axes
#' A string containing the output dimension(s)
#' to apply a bias to. Each character in the `bias_axes` string
#' should correspond to a character in the output portion
#' of the `equation` string.
#'
#' @param kernel_initializer
#' Initializer for the `kernel` weights matrix.
#'
#' @param bias_initializer
#' Initializer for the bias vector.
#'
#' @param kernel_regularizer
#' Regularizer function applied to the `kernel` weights
#' matrix.
#'
#' @param bias_regularizer
#' Regularizer function applied to the bias vector.
#'
#' @param kernel_constraint
#' Constraint function applied to the `kernel` weights
#' matrix.
#'
#' @param bias_constraint
#' Constraint function applied to the bias vector.
#'
#' @param ...
#' Base layer keyword arguments, such as `name` and `dtype`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @export
#' @family core layers
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/EinsumDense>
#'
#' @tether keras.layers.EinsumDense
layer_einsum_dense <-
function (object, equation, output_shape, activation = NULL,
    bias_axes = NULL, kernel_initializer = "glorot_uniform",
    bias_initializer = "zeros", kernel_regularizer = NULL, bias_regularizer = NULL,
    kernel_constraint = NULL, bias_constraint = NULL, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape,
        output_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$EinsumDense, object, args)
}


#' Turns positive integers (indexes) into dense vectors of fixed size.
#'
#' @description
#' e.g. `rbind(4L, 20L)` \eqn{\rightarrow}{->} `rbind(c(0.25, 0.1), c(0.6, -0.2))`
#'
#' This layer can only be used on positive integer inputs of a fixed range.
#'
#' # Examples
#' ```{r}
#' model <- keras_model_sequential() |>
#'   layer_embedding(1000, 64)
#'
#' # The model will take as input an integer matrix of size (batch,input_length),
#' # and the largest integer (i.e. word index) in the input
#' # should be no larger than 999 (vocabulary size).
#' # Now model$output_shape is (NA, 10, 64), where `NA` is the batch
#' # dimension.
#'
#' input_array <- random_integer(shape = c(32, 10), minval = 0, maxval = 1000)
#' model |> compile('rmsprop', 'mse')
#' output_array <- model |> predict(input_array, verbose = 0)
#' dim(output_array)    # (32, 10, 64)
#' ```
#'
#' # Input Shape
#' 2D tensor with shape: `(batch_size, input_length)`.
#'
#' # Output Shape
#' 3D tensor with shape: `(batch_size, input_length, output_dim)`.
#'
#' @param input_dim
#' Integer. Size of the vocabulary,
#' i.e. maximum integer index + 1.
#'
#' @param output_dim
#' Integer. Dimension of the dense embedding.
#'
#' @param embeddings_initializer
#' Initializer for the `embeddings`
#' matrix (see `keras3::initializer_*`).
#'
#' @param embeddings_regularizer
#' Regularizer function applied to
#' the `embeddings` matrix (see `keras3::regularizer_*`).
#'
#' @param embeddings_constraint
#' Constraint function applied to
#' the `embeddings` matrix (see `keras3::constraint_*`).
#'
#' @param mask_zero
#' Boolean, whether or not the input value 0 is a special
#' "padding" value that should be masked out.
#' This is useful when using recurrent layers which
#' may take variable length input. If this is `TRUE`,
#' then all subsequent layers in the model need
#' to support masking or an exception will be raised.
#' If `mask_zero` is set to `TRUE`, as a consequence,
#' index 0 cannot be used in the vocabulary (`input_dim` should
#' equal size of vocabulary + 1).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family core layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/core_layers/embedding#embedding-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding>
#' @tether keras.layers.Embedding
layer_embedding <-
function (object, input_dim, output_dim, embeddings_initializer = "uniform",
    embeddings_regularizer = NULL, embeddings_constraint = NULL,
    mask_zero = FALSE, ...)
{
    args <- capture_args2(list(input_dim = as_integer, output_dim = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape, input_length = as_integer),
        ignore = "object")
    create_layer(keras$layers$Embedding, object, args)
}


#' Identity layer.
#'
#' @description
#' This layer should be used as a placeholder when no operation is to be
#' performed. The layer just returns its `inputs` argument as output.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family core layers
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Identity>
#' @tether keras.layers.Identity
layer_identity <-
function (object, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Identity, object, args)
}


#' Used to instantiate a Keras tensor.
#'
#' @description
#' A Keras tensor is a symbolic tensor-like object, which we augment with
#' certain attributes that allow us to build a Keras model just by knowing the
#' inputs and outputs of the model.
#'
#' For instance, if `a`, `b` and `c` are Keras tensors,
#' it becomes possible to do:
#' `model <- keras_model(input = c(a, b), output = c)`
#'
#' # Examples
#' ```{r}
#' # This is a logistic regression in Keras
#' input <- layer_input(shape=c(32))
#' output <- input |> layer_dense(16, activation='softmax')
#' model <- keras_model(input, output)
#' ```
#'
#' @returns
#' A Keras tensor.
#'
#' @param shape
#' A shape list (list of integers or `NULL` objects),
#' not including the batch size.
#' For instance, `shape = c(32)` indicates that the expected input
#' will be batches of 32-dimensional vectors. Elements of this list
#' can be `NULL` or `NA`; `NULL`/`NA` elements represent dimensions where the shape
#' is not known and may vary (e.g. sequence length).
#'
#' @param batch_size
#' Optional static batch size (integer).
#'
#' @param dtype
#' The data type expected by the input, as a string
#' (e.g. `"float32"`, `"int32"`...)
#'
#' @param sparse
#' A boolean specifying whether the expected input will be sparse
#' tensors. Note that, if `sparse` is `FALSE`, sparse tensors can still
#' be passed into the input - they will be densified with a default
#' value of 0. This feature is only supported with the TensorFlow
#' backend. Defaults to `FALSE`.
#'
#' @param name
#' Optional name string for the layer.
#' Should be unique in a model (do not reuse the same name twice).
#' It will be autogenerated if it isn't provided.
#'
#' @param tensor
#' Optional existing tensor to wrap into the `Input` layer.
#' If set, the layer will use this tensor rather
#' than creating a new placeholder tensor.
#'
#' @param batch_shape
#' Shape, including the batch dim.
#'
#' @export
#' @family input core layers
#' @family core layers
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Input>
#'
#' @tether keras.layers.Input
layer_input <-
function (shape = NULL, batch_size = NULL, dtype = NULL, sparse = NULL,
    batch_shape = NULL, name = NULL, tensor = NULL)
{
    args <- capture_args2(list(shape = normalize_shape, batch_size = as_integer,
        input_shape = normalize_shape, batch_input_shape = normalize_shape,
        batch_shape = normalize_shape))
    do.call(keras$layers$Input, args)
}


#' Wraps arbitrary expressions as a `Layer` object.
#'
#' @description
#' The `layer_lambda()` layer exists so that arbitrary expressions can be used
#' as a `Layer` when constructing Sequential
#' and Functional API models. `Lambda` layers are best suited for simple
#' operations or quick experimentation. For more advanced use cases,
#' prefer writing new subclasses of `Layer` using [`new_layer_class()`].
#'
#'
#' # Examples
#' ```{r}
#' # add a x -> x^2 layer
#' model <- keras_model_sequential()
#' model |> layer_lambda(\(x) x^2)
#' ```
#'
#' @param f
#' The function to be evaluated. Takes input tensor as first
#' argument.
#'
#' @param output_shape
#' Expected output shape from function. This argument
#' can usually be inferred if not explicitly provided.
#' Can be a list or function. If a list, it only specifies
#' the first dimension onward; sample dimension is assumed
#' either the same as the input:
#' `output_shape = c(input_shape[1], output_shape)` or,
#' the input is `NULL` and the sample dimension is also `NULL`:
#' `output_shape = c(NA, output_shape)`.
#' If a function, it specifies the
#' entire shape as a function of the input shape:
#' `output_shape = f(input_shape)`.
#'
#' @param mask
#' Either `NULL` (indicating no masking) or a callable with the same
#' signature as the `compute_mask` layer method, or a tensor
#' that will be returned as output mask regardless
#' of what the input is.
#'
#' @param arguments
#' Optional named list of arguments to be passed to the
#' function.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family core layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/core_layers/lambda#lambda-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda>
#'
#' @tether keras.layers.Lambda
layer_lambda <-
function (object, f, output_shape = NULL, mask = NULL, arguments = NULL,
    ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape,
        output_shape = normalize_shape), ignore = "object")
    names(args)[match("f", names(args))] <- "function"
    create_layer(keras$layers$Lambda, object, args)
}


#' Masks a sequence by using a mask value to skip timesteps.
#'
#' @description
#' For each timestep in the input tensor (dimension #1 in the tensor),
#' if all values in the input tensor at that timestep
#' are equal to `mask_value`, then the timestep will be masked (skipped)
#' in all downstream layers (as long as they support masking).
#'
#' If any downstream layer does not support masking yet receives such
#' an input mask, an exception will be raised.
#'
#' # Examples
#' Consider an array `x` of shape `c(samples, timesteps, features)`,
#' to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
#' lack data for these timesteps. You can:
#'
#' - Set `x[, 3, ] <- 0.` and `x[, 5, ] <- 0.`
#' - Insert a `layer_masking()` layer with `mask_value = 0.` before the LSTM layer:
#'
#' ```{r}
#' c(samples, timesteps, features) %<-% c(32, 10, 8)
#' inputs <- c(samples, timesteps, features) %>% { array(runif(prod(.)), dim = .) }
#' inputs[, 3, ] <- 0
#' inputs[, 5, ] <- 0
#'
#' model <- keras_model_sequential() %>%
#'   layer_masking(mask_value = 0) %>%
#'   layer_lstm(32)
#'
#' output <- model(inputs)
#' # The time step 3 and 5 will be skipped from LSTM calculation.
#' ```
#'
#' # Note
#' in the Keras masking convention, a masked timestep is denoted by
#' a mask value of `FALSE`, while a non-masked (i.e. usable) timestep
#' is denoted by a mask value of `TRUE`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param mask_value
#' see description
#'
#' @export
#' @family core layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/core_layers/masking#masking-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking>
#' @tether keras.layers.Masking
layer_masking <-
function (object, mask_value = 0, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Masking, object, args)
}
