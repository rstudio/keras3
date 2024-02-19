

#' Cast a tensor to the desired dtype.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' (x <- op_arange(4))
#' op_cast(x, dtype = "float16")
#' ```
#'
#' @returns
#' A tensor of the specified `dtype`.
#'
#' @param x
#' A tensor or variable.
#'
#' @param dtype
#' The target type.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#cast-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cast>
#' @tether keras.ops.cast
op_cast <-
function (x, dtype)
keras$ops$cast(x, dtype)


#' Conditionally applies `true_fn` or `false_fn`.
#'
#' @returns
#' The output of either `true_fn` or `false_fn` depending on pred.
#'
#' @param pred
#' Boolean scalar type
#'
#' @param true_fn
#' Callable returning the output for the `pred == TRUE` case.
#'
#' @param false_fn
#' Callable returning the output for the `pred == FALSE` case.
#'
#' @details
#'
#' # Examples
#' ```{r}
#' fn <- tensorflow::tf_function(function(x) {
#'   op_cond(x > 0,
#'     true_fn = \() x + 1,
#'     false_fn = \() x - 1)
#' })
#'
#' fn(tensorflow::as_tensor(1))
#' fn(tensorflow::as_tensor(-1))
#' #
#' # Conditional side-effect (print only, no return value).
#' file <- tempfile(fileext = ".txt")
#' fn <- tensorflow::tf_function(function(epochs) {
#'   op_fori_loop(
#'     0, epochs,
#'     body_fun = \(epoch, state) {
#'       op_cond(epoch %% 20 == 0,
#'               \() {
#'                 tensorflow::tf$print(
#'                   "epoch:", epoch,
#'                   output_stream = paste0("file://", file))
#'                 NULL
#'               },
#'               \() {NULL})
#'       state
#'     },
#'     init_val = tensorflow::as_tensor(0))
#' })
#'
#' fn(tensorflow::as_tensor(100))
#'
#' readLines(file)
#'
#' # cleanup
#' unlink(file)
#' ```
#' @export
#' @family core ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cond>
#' @tether keras.ops.cond
op_cond <-
function (pred, true_fn, false_fn)
keras$ops$cond(pred, true_fn, false_fn)


#' Convert a tensor to a NumPy array.
#'
#' @returns
#' A NumPy array.
#'
#' @param x
#' A tensor.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#converttonumpy-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/convert_to_numpy>
#' @tether keras.ops.convert_to_numpy
op_convert_to_numpy <-
function (x)
keras$ops$convert_to_numpy(x)


#' Convert an array to a tensor.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- array(c(1, 2, 3))
#' y <- op_convert_to_tensor(x)
#' y
#' ```
#'
#' @returns
#' A tensor of the specified `dtype`.
#'
#' @param x
#' An array.
#'
#' @param dtype
#' The target type.
#'
#' @param sparse
#' Whether to keep sparse tensors. `FALSE` will cause sparse
#' tensors to be densified. The default value of `NULL` means that
#' sparse tensors are kept only if the backend supports them.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + [op_array()]
#' + <https://keras.io/api/ops/core#converttotensor-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/convert_to_tensor>
#' @tether keras.ops.convert_to_tensor
op_convert_to_tensor <-
function (x, dtype = NULL, sparse = NULL)
keras$ops$convert_to_tensor(x, dtype, sparse)


#' For loop implementation.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' lower <- 0L
#' upper <- 10L
#' body_fun <- function(i, state) state + i
#' init_state <- 0L
#' final_state <- op_fori_loop(lower, upper, body_fun, init_state)
#' final_state
#' ```
#'
#' @returns
#' The final state after the loop.
#'
#' @param lower
#' The initial value of the loop variable.
#'
#' @param upper
#' The upper bound of the loop variable.
#'
#' @param body_fun
#' A callable that represents the loop body. Must take two
#' arguments: the loop variable and the loop state. The loop state
#' should be updated and returned by this function.
#'
#' @param init_val
#' The initial value of the loop state.
#'
#' @export
#' @family core ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/fori_loop>
#' @tether keras.ops.fori_loop
op_fori_loop <-
function (lower, upper, body_fun, init_val)
keras$ops$fori_loop(lower, upper, body_fun, init_val)


#' Check whether the given object is a tensor.
#'
#' @description
#'
#' # Note
#' This checks for backend specific tensors so passing a TensorFlow
#' tensor would return `FALSE` if your backend is PyTorch or JAX.
#'
#' @returns
#' `TRUE` if `x` is a tensor, otherwise `FALSE`.
#'
#' @param x
#' A variable.
#'
#' @export
#' @family core ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/is_tensor>
#' @tether keras.ops.is_tensor
op_is_tensor <-
function (x)
keras$ops$is_tensor(x)


#' Returns a tensor of shape `shape` where `indices` are set to `values`.
#'
#' @description
#' At a high level, this operation does `zeros[indices] = updates` and
#' returns the output. It is equivalent to:
#'
#' ```{r, eval = FALSE}
#' output <- op_scatter_update(op_zeros(shape), indices, values)
#' ```
#'
#' # Examples
#' ```{r}
#' indices <- rbind(c(1, 2), c(2, 2))
#' values <- op_array(c(1, 1))
#' op_scatter(indices, values, shape= c(2, 2))
#' ```
#'
#' @param indices
#' A tensor or list specifying
#' indices for the values in `values`.
#'
#' @param values
#' A tensor, the values to be set at `indices`.
#'
#' @param shape
#' Shape of the output tensor.
#'
#' @returns A tensor of shape `shape` where `indices` are set to `values`.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#scatter-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/scatter>
#' @tether keras.ops.scatter
op_scatter <-
function (indices, values, shape)
{
    args <- capture_args(list(indices = as_index, shape = normalize_shape))
    do.call(keras$ops$scatter, args)
}


#' Update inputs via updates at scattered (sparse) indices.
#'
#' @description
#' At a high level, this operation does `inputs[indices] = updates`.
#' Assume `inputs` is a tensor of shape `(D1, D2, ..., Dn)`, there are 2 main
#' usages of `scatter_update`.
#'
#' 1. `indices` is a 2D tensor of shape `(num_updates, n)`, where `num_updates`
#'     is the number of updates to perform, and `updates` is a 1D tensor of
#'     shape `(num_updates)`. For example, if `inputs` is `op_zeros(c(4, 4, 4))`,
#'     and we want to update `inputs[2, 3, 4]` and `inputs[1, 2, 4]` as 1, then
#'     we can use:
#'
#' ```{r}
#' inputs <- op_zeros(c(4, 4, 4))
#' indices <- rbind(c(2, 3, 4), c(1, 2, 4))
#' updates <- op_array(c(1, 1), "float32")
#' op_scatter_update(inputs, indices, updates)
#' ```
#'
#' 2 `indices` is a 2D tensor of shape `(num_updates, k)`, where `num_updates`
#'     is the number of updates to perform, and `k` (`k <= n`) is the size of
#'     each index in `indices`. `updates` is a `n - k`-D tensor of shape
#'     `(num_updates, inputs.shape[k:))`. For example, if
#'     `inputs = op_zeros(c(4, 4, 4))`, and we want to update `inputs[1, 2, ]`
#'     and `inputs[2, 3, ]` as `[1, 1, 1, 1]`, then `indices` would have shape
#'     `(num_updates, 2)` (`k = 2`), and `updates` would have shape
#'     `(num_updates, 4)` (`inputs.shape[2:] = 4`). See the code below:
#'
#' ```{r}
#' inputs <- op_zeros(c(4, 4, 4))
#' indices <- rbind(c(2, 3), c(3, 4))
#' updates <- op_array(rbind(c(1, 1, 1, 1), c(1, 1, 1, 1)), "float32")
#' op_scatter_update(inputs, indices, updates)
#' ```
#'
#' @returns
#' A tensor, has the same shape and dtype as `inputs`.
#'
#' @param inputs
#' A tensor, the tensor to be updated.
#'
#' @param indices
#' A tensor or list of shape `(N, inputs$ndim)`, specifying
#' indices to update. `N` is the number of indices to update, must be
#' equal to the first dimension of `updates`.
#'
#' @param updates
#' A tensor, the new values to be put to `inputs` at `indices`.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#scatterupdate-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/scatter_update>
#' @tether keras.ops.scatter_update
op_scatter_update <-
function (inputs, indices, updates)
{
    args <- capture_args(list(indices = as_index))
    do.call(keras$ops$scatter_update, args)
}


#' Gets the shape of the tensor input.
#'
#' @description
#'
#' # Note
#' On the TensorFlow backend, when `x` is a `tf.Tensor` with dynamic
#' shape, dimensions which are dynamic in the context of a compiled function
#' will have a `tf.Tensor` value instead of a static integer value.
#'
#' # Examples
#' ```{r}
#' x <- op_zeros(c(8, 12))
#' op_shape(x)
#' ```
#'
#' @returns
#' A list of integers or NULL values, indicating the shape of the input
#' tensor.
#'
#' @param x
#' A tensor. This function will try to access the `shape` attribute of
#' the input tensor.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#shape-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/shape>
#' @tether keras.ops.shape
op_shape <-
function (x)
{
    out <- keras$ops$shape(x)
    class(out) <- "keras_shape"
    out
}


#' Return a slice of an input tensor.
#'
#' @description
#' At a high level, this operation is an explicit replacement for array slicing
#' e.g. `inputs[start_indices:(start_indices + shape)]`.
#' Unlike slicing via brackets, this operation will accept tensor start
#' indices on all backends, which is useful when indices dynamically computed
#' via other tensor operations.
#'
#' ```{r}
#' (inputs <- op_arange(5*5) |> op_reshape(c(5, 5)))
#' start_indices <- c(3, 3)
#' shape <- c(2, 2)
#' op_slice(inputs, start_indices, shape)
#' ```
#'
#' @returns
#' A tensor, has the same shape and dtype as `inputs`.
#'
#' @param inputs
#' A tensor, the tensor to be sliced.
#'
#' @param start_indices
#' A list of length `inputs$ndim`, specifying
#' the starting indices for updating.
#'
#' @param shape
#' The full shape of the returned slice.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#slice-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/slice>
#'
#' @tether keras.ops.slice
op_slice <-
function (inputs, start_indices, shape)
{
    args <- capture_args(list(shape = normalize_shape, start_indices = as_index))
    do.call(keras$ops$slice, args)
}


#' Update an input by slicing in a tensor of updated values.
#'
#' @description
#' At a high level, this operation does
#' `inputs[start_indices: start_indices + updates.shape] = updates`.
#' Assume inputs is a tensor of shape `(D1, D2, ..., Dn)`,
#' `start_indices` must be a list of n integers, specifying the starting
#' indices. `updates` must have the same rank as `inputs`, and the size of each
#' dim must not exceed `Di - start_indices[i]`. For example, if we have 2D
#' inputs `inputs = op_zeros(c(5, 5))`, and we want to update the intersection
#' of last 2 rows and last 2 columns as 1, i.e.,
#' `inputs[4:5, 4:5] = op_ones(c(2, 2))`, then we can use the code below:
#'
#' ```{r}
#' inputs <- op_zeros(c(5, 5))
#' start_indices <- c(3, 3)
#' updates <- op_ones(c(2, 2))
#' op_slice_update(inputs, start_indices, updates)
#' ```
#'
#' @returns
#' A tensor, has the same shape and dtype as `inputs`.
#'
#' @param inputs
#' A tensor, the tensor to be updated.
#'
#' @param start_indices
#' A list of length `inputs$ndim`, specifying
#' the starting indices for updating.
#'
#' @param updates
#' A tensor, the new values to be put to `inputs` at `indices`.
#' `updates` must have the same rank as `inputs`.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#sliceupdate-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/slice_update>
#' @tether keras.ops.slice_update
op_slice_update <-
function (inputs, start_indices, updates)
{
    args <- capture_args(list(start_indices = as_index))
    do.call(keras$ops$slice_update, args)
}


#' Stops gradient computation.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' var <- op_convert_to_tensor(c(1, 2, 3), dtype="float32")
#' var <- op_stop_gradient(var)
#' ```
#'
#' @returns
#' The variable with gradient computation disabled.
#'
#' @param variable
#' A tensor variable for which the gradient
#' computation is to be disabled.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#stopgradient-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/stop_gradient>
#'
#' @tether keras.ops.stop_gradient
op_stop_gradient <-
function (variable)
keras$ops$stop_gradient(variable)


#' Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_array(rbind(c(1, 2),
#'                    c(3, 4)))
#' op_unstack(x, axis=1)
#' op_unstack(x, axis=2)
#' ```
#'
#'
#' ```{r}
#' all.equal(op_unstack(x),
#'           op_unstack(x, axis = 1))
#' all.equal(op_unstack(x, axis = -1),
#'           op_unstack(x, axis = 2))
#' # [array([1, 2)), array([3, 4))]
#' ```
#'
#' @returns
#' A list of tensors unpacked along the given axis.
#'
#' @param x
#' The input tensor.
#'
#' @param num
#' The length of the dimension axis. Automatically inferred
#' if `NULL`.
#'
#' @param axis
#' The axis along which to unpack.
#'
#' @export
#' @family core ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/unstack>
#'
#' @tether keras.ops.unstack
op_unstack <-
function (x, num = NULL, axis = 1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$unstack, args)
}


#' Parallel map of function `f` on the first axis of tensor(s) `elements`.
#'
#' @description
#' Schematically, `vectorized_map` implements the following,
#' in the case of a single tensor input `elements`:
#'
#' ```{r, eval = FALSE}
#' op_vectorized_map <- function(elements, f) {
#'   apply(elements, 1, f)
#' }
#' ```
#'
#' In the case of an iterable of tensors `elements`,
#' it implements the following:
#'
#' ```{r, eval = FALSE}
#' op_vectorized_map <- function(elements, f) {
#'     batch_size <- elements[[1]] |> shape() |> _[[1]]
#'     outputs <- vector("list", batch_size)
#'     outputs <- lapply(seq(batch_size), \(index) {
#'         f(lapply(elements, \(e) e[index, all_dims()]))
#'     }
#'     op_stack(outputs)
#' }
#' ```
#'
#' In this case, `function` is expected to take as input
#' a single list of tensor arguments.
#'
#'
#' ```{r}
#' (x <- op_arange(4*4) |> op_reshape(c(4,4)))
#' x |> op_vectorized_map(\(row) {row + 10})
#' list(x, x, x) |> op_vectorized_map(\(rows) Reduce(`+`, rows))
#' ```
#'
#' @param elements
#' see description
#'
#' @param f
#' A function taking either a tensor, or list of tensors.
#'
#' @returns A tensor, the result of mapping `f` across `elements.`
#' @export
#' @family core ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/vectorized_map>
#' @tether keras.ops.vectorized_map
op_vectorized_map <-
function (elements, f)
keras$ops$vectorized_map(f, elements)


#' While loop implementation.
#'
#' @description
#'
#' # Examples
#'
#' ```{r}
#' i <- 0
#' loop_vars <- list(i)
#'
#' # cond() must return a scalar bool
#' cond <- function(i) i < 10L
#'
#' # body must return same shape as loop_vars
#' body <- function(i) list(i + 1L)
#'
#' op_while_loop(cond, body, loop_vars)
#' ```
#'
#' ```{r}
#' x <- 0; y <- 1
#' cond <- \(x, y) x < 10
#' body <- \(x, y) list(x+1, y+1)
#' op_while_loop(cond, body, list(x, y))
#' ```
#'
#' @returns
#' A list of tensors, has the same shape and dtype as `loop_vars`.
#'
#' @param cond
#' A callable that represents the termination condition of the loop.
#' Must accept a `loop_vars` like structure as an argument. If
#'`loop_vars` is a tuple or unnamed list, each element of `loop_vars` will be
#' passed positionally to the callable.
#'
#' @param body
#' A callable that represents the loop body. Must accept a
#' `loop_vars` like structure as an argument, and return update value
#' with the same structure. If `loop_vars` is a tuple or unnamed list, each
#' element of `loop_vars` will be passed positionally to the callable.
#'
#' @param loop_vars
#' An arbitrary nested structure of tensor state to persist
#' across loop iterations.
#'
#' @param maximum_iterations
#' Optional maximum number of iterations of the while
#' loop to run. If provided, the `cond` output is AND-ed with an
#' additional condition ensuring the number of iterations executed is
#' no greater than `maximum_iterations`.
#'
#' @export
#' @family core ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#whileloop-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/while_loop>
#'
#' @tether keras.ops.while_loop
op_while_loop <-
function (cond, body, loop_vars, maximum_iterations = NULL)
keras$ops$while_loop(cond, body, loop_vars, maximum_iterations)


#' Computes the error function of `x`, element-wise.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-3, -2, -1, 0, 1))
#' op_erf(x)
#' # array([-0.99998 , -0.99532, -0.842701,  0.,  0.842701], dtype=float32)
#' ```
#'
#' @returns
#' A tensor with the same dtype as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family math ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/erf>
#' @tether keras.ops.erf
op_erf <-
function (x)
keras$ops$erf(x)


#' Expands the dimension of last axis into sequences of `sequence_length`.
#'
#' @description
#' Slides a window of size `sequence_length` over the last axis of the input
#' with a stride of `sequence_stride`, replacing the last axis with
#' `[num_sequences, sequence_length]` sequences.
#'
#' If the dimension along the last axis is N, the number of sequences can be
#' computed by:
#'
#' `num_sequences = 1 + (N - sequence_length) // sequence_stride`
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(1:6)
#' op_extract_sequences(x, 3, 2)
#' ```
#'
#' @returns
#' A tensor of sequences with shape `[..., num_sequences, sequence_length].`
#'
#' @param x
#' Input tensor.
#'
#' @param sequence_length
#' An integer representing the sequences length.
#'
#' @param sequence_stride
#' An integer representing the sequences hop size.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#extractsequences-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/extract_sequences>
#' @tether keras.ops.extract_sequences
op_extract_sequences <-
function (x, sequence_length, sequence_stride)
{
    args <- capture_args(list(sequence_length = as_integer,
        sequence_stride = as_integer))
    do.call(keras$ops$extract_sequences, args)
}


#' Computes the Fast Fourier Transform along last axis of input.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x = c(op_array(c(1., 2.)),
#'       op_array(c(0., 1.)))
#' op_fft(x)
#' ```
#'
#' @returns
#' A list containing two tensors - the real and imaginary parts of the
#' output tensor.
#'
#' @param x
#' list of the real and imaginary parts of the input tensor. Both
#' tensors provided should be of floating type.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/fft#fft-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/fft>
#' @tether keras.ops.fft
op_fft <-
function (x)
keras$ops$fft(x)


#' Computes the 2D Fast Fourier Transform along the last two axes of input.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- c(op_array(rbind(c(1, 2),
#'                      c(2, 1))),
#'        op_array(rbind(c(0, 1),
#'                      c(1, 0))))
#' op_fft2(x)
#' ```
#'
#' @returns
#' A list containing two tensors - the real and imaginary parts of the
#' output.
#'
#' @param x
#' list of the real and imaginary parts of the input tensor. Both
#' tensors provided should be of floating type.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/fft#fft2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/fft2>
#' @tether keras.ops.fft2
op_fft2 <-
function (x)
keras$ops$fft2(x)


#' Checks if the targets are in the top-k predictions.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' targets <- op_array(c(2, 5, 3), "int32")
#' predictions <- op_array(dtype = "float32", rbind(
#'   c(0.1, 0.4, 0.6, 0.9, 0.5),
#'   c(0.1, 0.7, 0.9, 0.8, 0.3),
#'   c(0.1, 0.6, 0.9, 0.9, 0.5)
#' ))
#' op_in_top_k(targets, predictions, k = 3L)
#' ```
#'
#' @returns
#' A boolean tensor of the same shape as `targets`, where each element
#' indicates whether the corresponding target is in the top-k predictions.
#'
#' @param targets
#' A tensor of true labels.
#'
#' @param predictions
#' A tensor of predicted labels.
#'
#' @param k
#' An integer representing the number of predictions to consider.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#intopk-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/in_top_k>
#' @tether keras.ops.in_top_k
op_in_top_k <-
function (targets, predictions, k)
{
    args <- capture_args(list(k = as_integer))
    do.call(keras$ops$in_top_k, args)
}


#' Inverse real-valued Fast Fourier transform along the last axis.
#'
#' @description
#' Computes the inverse 1D Discrete Fourier Transform of a real-valued signal
#' over the inner-most dimension of input.
#'
#' The inner-most dimension of the input is assumed to be the result of RFFT:
#' the `fft_length / 2 + 1` unique components of the DFT of a real-valued
#' signal. If `fft_length` is not provided, it is computed from the size of the
#' inner-most dimension of the input `(fft_length = 2 * (inner - 1))`. If the
#' FFT length used to compute is odd, it should be provided since it cannot
#' be inferred properly.
#'
#' Along the axis IRFFT is computed on, if `fft_length / 2 + 1` is smaller than
#' the corresponding dimension of the input, the dimension is cropped. If it is
#' larger, the dimension is padded with zeros.
#'
#' # Examples
#'
#' ```{r, comment = "#>"}
#' real <- op_array(c(0, 1, 2, 3, 4))
#' imag <- op_array(c(0, 1, 2, 3, 4))
#' op_irfft(c(real, imag))
#'
#' all.equal(op_irfft(op_rfft(real, 5), 5), real)
#' ```
#'
#' @returns
#' A tensor containing the inverse real-valued Fast Fourier Transform
#' along the last axis of `x`.
#'
#' @param x
#' List of the real and imaginary parts of the input tensor. Both
#' tensors in the list should be of floating type.
#'
#' @param fft_length
#' An integer representing the number of the fft length. If not
#' specified, it is inferred from the length of the last axis of `x`.
#' Defaults to `NULL`.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/fft#irfft-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/irfft>
#' @tether keras.ops.irfft
op_irfft <-
function (x, fft_length = NULL)
{
    args <- capture_args(list(fft_length = as_integer))
    do.call(keras$ops$irfft, args)
}


#' Inverse Short-Time Fourier Transform along the last axis of the input.
#'
#' @description
#' To reconstruct an original waveform, the parameters should be the same in
#' `stft`.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(0, 1, 2, 3, 4))
#' op_istft(op_stft(x, 1, 1, 1), 1, 1, 1)
#' # array([0.0, 1.0, 2.0, 3.0, 4.0])
#' ```
#'
#' @returns
#' A tensor containing the inverse Short-Time Fourier Transform along the
#' last axis of `x`.
#'
#' @param x
#' Tuple of the real and imaginary parts of the input tensor. Both
#' tensors in the list should be of floating type.
#'
#' @param sequence_length
#' An integer representing the sequence length.
#'
#' @param sequence_stride
#' An integer representing the sequence hop size.
#'
#' @param fft_length
#' An integer representing the size of the FFT that produced
#' `stft`.
#'
#' @param length
#' An integer representing the output is clipped to exactly length.
#' If not specified, no padding or clipping take place. Defaults to
#' `NULL`.
#'
#' @param window
#' A string, a tensor of the window or `NULL`. If `window` is a
#' string, available values are `"hann"` and `"hamming"`. If `window`
#' is a tensor, it will be used directly as the window and its length
#' must be `sequence_length`. If `window` is `NULL`, no windowing is
#' used. Defaults to `"hann"`.
#'
#' @param center
#' Whether `x` was padded on both sides so that the t-th sequence
#' is centered at time `t * sequence_stride`. Defaults to `TRUE`.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/fft#istft-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/istft>
#' @tether keras.ops.istft
op_istft <-
function (x, sequence_length, sequence_stride, fft_length, length = NULL,
    window = "hann", center = TRUE)
{
    args <- capture_args(list(sequence_length = as_integer,
        sequence_stride = as_integer, fft_length = as_integer,
        length = as_integer, x = tuple))
    do.call(keras$ops$istft, args)
}


#' Computes the logarithm of sum of exponentials of elements in a tensor.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(1, 2, 3))
#' op_logsumexp(x)
#' ```
#'
#' @returns
#' A tensor containing the logarithm of the sum of exponentials of
#' elements in `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' An integer or a list of integers specifying the axis/axes
#' along which to compute the sum. If `NULL`, the sum is computed
#' over all elements. Defaults to`NULL`.
#'
#' @param keepdims
#' A boolean indicating whether to keep the dimensions of
#' the input tensor when computing the sum. Defaults to`FALSE`.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#logsumexp-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logsumexp>
#'
#' @tether keras.ops.logsumexp
op_logsumexp <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$logsumexp, args)
}


#' Computes the QR decomposition of a tensor.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(rbind(c(1, 2), c(3, 4), c(5, 6)))
#' op_qr(x)
#' c(q, r) %<-% op_qr(x)
#' ```
#'
#' @returns
#' A list containing two tensors. The first tensor of shape `(..., M, K)`
#' is the orthogonal matrix `q` and the second tensor of shape
#' (..., K, N)` is the upper triangular matrix `r`, where `K = min(M, N)`.
#'
#' @param x
#' Input tensor of shape `(..., M, N)`.
#'
#' @param mode
#' A string specifying the mode of the QR decomposition.
#' - 'reduced': Returns the reduced QR decomposition. (default)
#' - 'complete': Returns the complete QR decomposition.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#qr-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/qr>
#' @tether keras.ops.qr
op_qr <-
function (x, mode = "reduced")
keras$ops$qr(x, mode)


#' @export
py_to_r.tensorflow.python.ops.gen_linalg_ops.Qr <- function(x) {
  x <- py_eval("tuple")(x)
  names(x) <- c("q", "r")
  x
}


#' Real-valued Fast Fourier Transform along the last axis of the input.
#'
#' @description
#' Computes the 1D Discrete Fourier Transform of a real-valued signal over the
#' inner-most dimension of input.
#'
#' Since the Discrete Fourier Transform of a real-valued signal is
#' Hermitian-symmetric, RFFT only returns the `fft_length / 2 + 1` unique
#' components of the FFT: the zero-frequency term, followed by the
#' `fft_length / 2` positive-frequency terms.
#'
#' Along the axis RFFT is computed on, if `fft_length` is smaller than the
#' corresponding dimension of the input, the dimension is cropped. If it is
#' larger, the dimension is padded with zeros.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(0, 1, 2, 3, 4))
#' op_rfft(x)
#' ```
#'
#' ```{r}
#' op_rfft(x, 3)
#' ```
#'
#' @returns
#' A list containing two tensors - the real and imaginary parts of the
#' output.
#'
#' @param x
#' Input tensor.
#'
#' @param fft_length
#' An integer representing the number of the fft length. If not
#' specified, it is inferred from the length of the last axis of `x`.
#' Defaults to `NULL`.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/fft#rfft-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/rfft>
#'
#' @tether keras.ops.rfft
op_rfft <-
function (x, fft_length = NULL)
{
    args <- capture_args(list(fft_length = as_integer))
    do.call(keras$ops$rfft, args)
}


#' Computes reciprocal of square root of x element-wise.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(1, 10, 100))
#' op_rsqrt(x)
#' # array([1, 0.31622776, 0.1], dtype=float32)
#' ```
#'
#' @returns
#' A tensor with the same dtype as `x`.
#'
#' @param x
#' input tensor
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#rsqrt-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/rsqrt>
#'
#' @tether keras.ops.rsqrt
op_rsqrt <-
function (x)
keras$ops$rsqrt(x)


#' Computes the max of segments in a tensor.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' data <- op_convert_to_tensor(c(1, 2, 10, 20, 100, 200))
#' segment_ids <- op_array(c(1, 1, 2, 2, 3, 3), "int32")
#' num_segments <- 3
#' op_segment_max(data, segment_ids, num_segments)
#' # array([2, 20, 200], dtype=int32)
#' ```
#'
#' @returns
#' A tensor containing the max of segments, where each element
#' represents the max of the corresponding segment in `data`.
#'
#' @param data
#' Input tensor.
#'
#' @param segment_ids
#' A 1-D tensor containing segment indices for each
#' element in `data`.
#'
#' @param num_segments
#' An integer representing the total number of
#' segments. If not specified, it is inferred from the maximum
#' value in `segment_ids`.
#'
#' @param sorted
#' A boolean indicating whether `segment_ids` is sorted.
#' Defaults to`FALSE`.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#segmentmax-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/segment_max>
#' @tether keras.ops.segment_max
op_segment_max <-
function (data, segment_ids, num_segments = NULL, sorted = FALSE)
{
    args <- capture_args(list(segment_ids = as_index, num_segments = as_integer))
    do.call(keras$ops$segment_max, args)
}


#' Computes the sum of segments in a tensor.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' data <- op_array(c(1, 2, 10, 20, 100, 200))
#' segment_ids <- op_array(c(1, 1, 2, 2, 3, 3), "int32")
#' num_segments <- 3
#' op_segment_sum(data, segment_ids, num_segments)
#' ```
#'
#' @returns
#' A tensor containing the sum of segments, where each element
#' represents the sum of the corresponding segment in `data`.
#'
#' @param data
#' Input tensor.
#'
#' @param segment_ids
#' A 1-D tensor containing segment indices for each
#' element in `data`.
#'
#' @param num_segments
#' An integer representing the total number of
#' segments. If not specified, it is inferred from the maximum
#' value in `segment_ids`.
#'
#' @param sorted
#' A boolean indicating whether `segment_ids` is sorted.
#' Defaults to`FALSE`.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#segmentsum-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/segment_sum>
#' @tether keras.ops.segment_sum
op_segment_sum <-
function (data, segment_ids, num_segments = NULL, sorted = FALSE)
{
    args <- capture_args(list(segment_ids = as_index, num_segments = as_integer))
    do.call(keras$ops$segment_sum, args)
}


#' Solves a linear system of equations given by `a x = b`.
#'
#' @description
#' Solves for `x` in the equation `a %*% x == b`.
#'
#' # Examples
#' ```{r}
#' a <- op_array(c(1, 2, 4, 5), dtype="float32") |> op_reshape(c(2, 2))
#' b <- op_array(c(2, 4, 8, 10), dtype="float32") |> op_reshape(c(2, 2))
#' op_solve(a, b)
#' ```
#'
#' @returns
#' A tensor of shape `(..., M)` or `(..., M, N)` representing the solution
#' of the linear system. Returned shape is identical to `b`.
#'
#' @param a
#' A tensor of shape `(..., M, M)` representing the coefficients matrix.
#'
#' @param b
#' A tensor of shape `(..., M)` or `(..., M, N)` represeting the
#' right-hand side or "dependent variable" matrix.
#'
#' @export
#' @family math ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/solve>
#'
#' @tether keras.ops.solve
op_solve <-
function (a, b)
keras$ops$solve(a, b)


#' Short-Time Fourier Transform along the last axis of the input.
#'
#' @description
#' The STFT computes the Fourier transform of short overlapping windows of the
#' input. This giving frequency components of the signal as they change over
#' time.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(0, 1, 2, 3, 4))
#' op_stft(x, 3, 2, 3)
#' ```
#'
#' @returns
#' A list containing two tensors - the real and imaginary parts of the
#' STFT output.
#'
#' @param x
#' Input tensor.
#'
#' @param sequence_length
#' An integer representing the sequence length.
#'
#' @param sequence_stride
#' An integer representing the sequence hop size.
#'
#' @param fft_length
#' An integer representing the size of the FFT to apply. If not
#' specified, uses the smallest power of 2 enclosing `sequence_length`.
#'
#' @param window
#' A string, a tensor of the window or `NULL`. If `window` is a
#' string, available values are `"hann"` and `"hamming"`. If `window`
#' is a tensor, it will be used directly as the window and its length
#' must be `sequence_length`. If `window` is `NULL`, no windowing is
#' used. Defaults to `"hann"`.
#'
#' @param center
#' Whether to pad `x` on both sides so that the t-th sequence is
#' centered at time `t * sequence_stride`. Otherwise, the t-th sequence
#' begins at time `t * sequence_stride`. Defaults to `TRUE`.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/fft#stft-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/stft>
#'
#' @tether keras.ops.stft
op_stft <-
function (x, sequence_length, sequence_stride, fft_length, window = "hann",
    center = TRUE)
{
    args <- capture_args(list(sequence_length = as_integer,
        sequence_stride = as_integer, fft_length = as_integer))
    do.call(keras$ops$stft, args)
}


#' Finds the top-k values and their indices in a tensor.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(5, 2, 7, 1, 9, 3), "int32")
#' op_top_k(x, k = 3)
#' ```
#'
#' ```{r}
#' c(values, indices) %<-% op_top_k(x, k = 3)
#' values
#' indices
#' ```
#'
#' @returns
#' A list containing two tensors. The first tensor contains the
#' top-k values, and the second tensor contains the indices of the
#' top-k values in the input tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param k
#' An integer representing the number of top elements to retrieve.
#'
#' @param sorted
#' A boolean indicating whether to sort the output in
#' descending order. Defaults to`TRUE`.
#'
#' @export
#' @family math ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/core#topk-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/top_k>
#' @tether keras.ops.top_k
op_top_k <-
function (x, k, sorted = TRUE)
{
    args <- capture_args(list(k = as_integer))
    do.call(keras$ops$top_k, args)
}


#' @export
py_to_r.tensorflow.python.ops.gen_nn_ops.TopKV2 <- function(x) {
  x <- py_eval("tuple")(x)
  names(x) <- c("values", "indices")
  x
}


#' Average pooling operation.
#'
#' @returns
#' A tensor of rank N+2, the result of the average pooling operation.
#'
#' @param inputs
#' Tensor of rank N+2. `inputs` has shape
#' `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
#' `data_format = "channels_last"`, or
#' `(batch_size, num_channels) + inputs_spatial_shape` if
#' `data_format = "channels_first"`. Pooling happens over the spatial
#' dimensions only.
#'
#' @param pool_size
#' int or tuple/list of integers of size
#' `len(inputs_spatial_shape)`, specifying the size of the pooling
#' window for each spatial dimension of the input tensor. If
#' `pool_size` is int, then every spatial dimension shares the same
#' `pool_size`.
#'
#' @param strides
#' int or tuple/list of integers of size
#' `len(inputs_spatial_shape)`. The stride of the sliding window for
#' each spatial dimension of the input tensor. If `strides` is int,
#' then every spatial dimension shares the same `strides`.
#'
#' @param padding
#' string, either `"valid"` or `"same"`. `"valid"` means no
#' padding is applied, and `"same"` results in padding evenly to the
#' left/right or up/down of the input such that output has the
#' same height/width dimension as the input when `strides = 1`.
#'
#' @param data_format
#' A string, either `"channels_last"` or `"channels_first"`.
#' `data_format` determines the ordering of the dimensions in the
#' inputs. If `data_format = "channels_last"`, `inputs` is of shape
#' `(batch_size, ..., channels)` while if
#' `data_format = "channels_first"`, `inputs` is of shape
#' `(batch_size, channels, ...)`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#averagepool-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/average_pool>
#' @tether keras.ops.average_pool
op_average_pool <-
function (inputs, pool_size, strides = NULL, padding = "valid",
    data_format = NULL)
{
    args <- capture_args(list(pool_size = as_integer, strides = as_integer))
    do.call(keras$ops$average_pool, args)
}


#' Computes binary cross-entropy loss between target and output tensor.
#'
#' @description
#' The binary cross-entropy loss is commonly used in binary
#' classification tasks where each input sample belongs to one
#' of the two classes. It measures the dissimilarity between the
#' target and output probabilities or logits.
#'
#' # Examples
#' ```{r}
#' target <- op_array(c(0, 1, 1, 0))
#' output <- op_array(c(0.1, 0.9, 0.8, 0.2))
#' op_binary_crossentropy(target, output)
#' ```
#'
#' @returns
#' Integer tensor: The computed binary cross-entropy loss between
#' `target` and `output`.
#'
#' @param target
#' The target tensor representing the true binary labels.
#' Its shape should match the shape of the `output` tensor.
#'
#' @param output
#' The output tensor representing the predicted probabilities
#' or logits. Its shape should match the shape of the
#' `target` tensor.
#'
#' @param from_logits
#' (optional) Whether `output` is a tensor of logits or
#' probabilities.
#' Set it to `TRUE` if `output` represents logits; otherwise,
#' set it to `FALSE` if `output` represents probabilities.
#' Defaults to `FALSE`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#binarycrossentropy-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/binary_crossentropy>
#' @tether keras.ops.binary_crossentropy
op_binary_crossentropy <-
function (target, output, from_logits = FALSE)
keras$ops$binary_crossentropy(target, output, from_logits)


#' Computes categorical cross-entropy loss between target and output tensor.
#'
#' @description
#' The categorical cross-entropy loss is commonly used in multi-class
#' classification tasks where each input sample can belong to one of
#' multiple classes. It measures the dissimilarity
#' between the target and output probabilities or logits.
#'
#' # Examples
#' ```{r}
#' target <- op_array(rbind(c(1, 0, 0),
#'                         c(0, 1, 0),
#'                         c(0, 0, 1)))
#' output <- op_array(rbind(c(0.9, 0.05, 0.05),
#'                         c(0.1, 0.8, 0.1),
#'                         c(0.2, 0.3, 0.5)))
#' op_categorical_crossentropy(target, output)
#' ```
#'
#' @returns
#' Integer tensor: The computed categorical cross-entropy loss between
#' `target` and `output`.
#'
#' @param target
#' The target tensor representing the true categorical labels.
#' Its shape should match the shape of the `output` tensor
#' except for the last dimension.
#'
#' @param output
#' The output tensor representing the predicted probabilities
#' or logits. Its shape should match the shape of the `target`
#' tensor except for the last dimension.
#'
#' @param from_logits
#' (optional) Whether `output` is a tensor of logits or
#' probabilities.
#' Set it to `TRUE` if `output` represents logits; otherwise,
#' set it to `FALSE` if `output` represents probabilities.
#' Defaults to `FALSE`.
#'
#' @param axis
#' (optional) The axis along which the categorical cross-entropy
#' is computed.
#' Defaults to `-1`, which corresponds to the last dimension of
#' the tensors.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#categoricalcrossentropy-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/categorical_crossentropy>
#' @tether keras.ops.categorical_crossentropy
op_categorical_crossentropy <-
function (target, output, from_logits = FALSE, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$categorical_crossentropy, args)
}


#' General N-D convolution.
#'
#' @description
#' This ops supports 1D, 2D and 3D convolution.
#'
#' @returns
#' A tensor of rank N+2, the result of the conv operation.
#'
#' @param inputs
#' Tensor of rank N+2. `inputs` has shape
#' `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
#' `data_format = "channels_last"`, or
#' `(batch_size, num_channels) + inputs_spatial_shape` if
#' `data_format = "channels_first"`.
#'
#' @param kernel
#' Tensor of rank N+2. `kernel` has shape
#' `(kernel_spatial_shape, num_input_channels, num_output_channels)`.
#' `num_input_channels` should match the number of channels in
#' `inputs`.
#'
#' @param strides
#' int or int tuple/list of `len(inputs_spatial_shape)`,
#' specifying the strides of the convolution along each spatial
#' dimension. If `strides` is int, then every spatial dimension shares
#' the same `strides`.
#'
#' @param padding
#' string, either `"valid"` or `"same"`. `"valid"` means no
#' padding is applied, and `"same"` results in padding evenly to the
#' left/right or up/down of the input such that output has the
#' same height/width dimension as the input when `strides = 1`.
#'
#' @param data_format
#' A string, either `"channels_last"` or `"channels_first"`.
#' `data_format` determines the ordering of the dimensions in the
#' inputs. If `data_format = "channels_last"`, `inputs` is of shape
#' `(batch_size, ..., channels)` while if
#' `data_format = "channels_first"`, `inputs` is of shape
#' `(batch_size, channels, ...)`.
#'
#' @param dilation_rate
#' int or int tuple/list of `len(inputs_spatial_shape)`,
#' specifying the dilation rate to use for dilated convolution. If
#' `dilation_rate` is int, then every spatial dimension shares
#' the same `dilation_rate`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#conv-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/conv>
#' @tether keras.ops.conv
op_conv <-
function (inputs, kernel, strides = 1L, padding = "valid", data_format = NULL,
    dilation_rate = 1L)
{
    args <- capture_args(list(strides = as_integer, dilation_rate = as_integer))
    do.call(keras$ops$conv, args)
}


#' General N-D convolution transpose.
#'
#' @description
#' Also known as de-convolution. This ops supports 1D, 2D and 3D convolution.
#'
#' @returns
#' A tensor of rank N+2, the result of the conv operation.
#'
#' @param inputs
#' Tensor of rank N+2. `inputs` has shape
#' `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
#' `data_format = "channels_last"`, or
#' `(batch_size, num_channels) + inputs_spatial_shape` if
#' `data_format = "channels_first"`.
#'
#' @param kernel
#' Tensor of rank N+2. `kernel` has shape
#' `[kernel_spatial_shape, num_output_channels, num_input_channels],`
#' `num_input_channels` should match the number of channels in
#' `inputs`.
#'
#' @param strides
#' int or int tuple/list of `len(inputs_spatial_shape)`,
#' specifying the strides of the convolution along each spatial
#' dimension. If `strides` is int, then every spatial dimension shares
#' the same `strides`.
#'
#' @param padding
#' string, either `"valid"` or `"same"`. `"valid"` means no
#' padding is applied, and `"same"` results in padding evenly to the
#' left/right or up/down of the input such that output has the
#' same height/width dimension as the input when `strides = 1`.
#'
#' @param output_padding
#' int or int tuple/list of `len(inputs_spatial_shape)`,
#' specifying the amount of padding along the height and width of
#' the output tensor. Can be a single integer to specify the same
#' value for all spatial dimensions. The amount of output padding
#' along a given dimension must be lower than the stride along that
#' same dimension. If set to `NULL` (default), the output shape is
#' inferred.
#'
#' @param data_format
#' A string, either `"channels_last"` or `"channels_first"`.
#' `data_format` determines the ordering of the dimensions in the
#' inputs. If `data_format = "channels_last"`, `inputs` is of shape
#' `(batch_size, ..., channels)` while if
#' `data_format = "channels_first"`, `inputs` is of shape
#' `(batch_size, channels, ...)`.
#'
#' @param dilation_rate
#' int or int tuple/list of `len(inputs_spatial_shape)`,
#' specifying the dilation rate to use for dilated convolution. If
#' `dilation_rate` is int, then every spatial dimension shares
#' the same `dilation_rate`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#convtranspose-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/conv_transpose>
#' @tether keras.ops.conv_transpose
op_conv_transpose <-
function (inputs, kernel, strides, padding = "valid", output_padding = NULL,
    data_format = NULL, dilation_rate = 1L)
{
    args <- capture_args(list(strides = as_integer, output_padding = as_integer,
        dilation_rate = as_integer))
    do.call(keras$ops$conv_transpose, args)
}


#' General N-D depthwise convolution.
#'
#' @description
#' This ops supports 1D and 2D depthwise convolution.
#'
#' @returns
#' A tensor of rank N+2, the result of the depthwise conv operation.
#'
#' @param inputs
#' Tensor of rank N+2. `inputs` has shape
#' `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
#' `data_format = "channels_last"`, or
#' `(batch_size, num_channels) + inputs_spatial_shape` if
#' `data_format = "channels_first"`.
#'
#' @param kernel
#' Tensor of rank N+2. `kernel` has shape
#' `[kernel_spatial_shape, num_input_channels, num_channels_multiplier],`
#' `num_input_channels` should match the number of channels in
#' `inputs`.
#'
#' @param strides
#' int or int tuple/list of `len(inputs_spatial_shape)`,
#' specifying the strides of the convolution along each spatial
#' dimension. If `strides` is int, then every spatial dimension shares
#' the same `strides`.
#'
#' @param padding
#' string, either `"valid"` or `"same"`. `"valid"` means no
#' padding is applied, and `"same"` results in padding evenly to the
#' left/right or up/down of the input such that output has the
#' same height/width dimension as the input when `strides = 1`.
#'
#' @param data_format
#' A string, either `"channels_last"` or `"channels_first"`.
#' `data_format` determines the ordering of the dimensions in the
#' inputs. If `data_format = "channels_last"`, `inputs` is of shape
#' `(batch_size, ..., channels)` while if
#' `data_format = "channels_first"`, `inputs` is of shape
#' `(batch_size, channels, ...)`.
#'
#' @param dilation_rate
#' int or int tuple/list of `len(inputs_spatial_shape)`,
#' specifying the dilation rate to use for dilated convolution. If
#' `dilation_rate` is int, then every spatial dimension shares
#' the same `dilation_rate`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#depthwiseconv-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/depthwise_conv>
#' @tether keras.ops.depthwise_conv
op_depthwise_conv <-
function (inputs, kernel, strides = 1L, padding = "valid", data_format = NULL,
    dilation_rate = 1L)
{
    args <- capture_args(list(strides = as_integer, dilation_rate = as_integer))
    do.call(keras$ops$depthwise_conv, args)
}


#' Exponential Linear Unit activation function.
#'
#' @description
#' It is defined as:
#'
#' `f(x) =  alpha * (exp(x) - 1.) for x < 0`, `f(x) = x for x >= 0`.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-1., 0., 1.))
#' op_elu(x)
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param alpha
#' A scalar, slope of positive section. Defaults to `1.0`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#elu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/elu>
#' @tether keras.ops.elu
op_elu <-
function (x, alpha = 1)
keras$ops$elu(x, alpha)


#' Gaussian Error Linear Unit (GELU) activation function.
#'
#' @description
#' If `approximate` is `TRUE`, it is defined as:
#' `f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
#'
#' Or if `approximate` is `FALSE`, it is defined as:
#' `f(x) = x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`,
#' where `P(X) ~ N(0, 1)`.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-1., 0., 1.))
#' op_gelu(x)
#' op_gelu(x, FALSE)
#' ```
#'
#'
#' ```{r op-gelu-plot}
#' x <- seq(-5, 5, .1)
#' plot(x, op_gelu(x),
#'      type = "l", #, frame.plot = FALSE,
#'      panel.first = grid())
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param approximate
#' Approximate version of GELU activation. Defaults to `TRUE`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#gelu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/gelu>
#' @tether keras.ops.gelu
op_gelu <-
function (x, approximate = TRUE)
keras$ops$gelu(x, approximate)


#'
#' Hard sigmoid activation function.
#'
#' @description
#' It is defined as:
#'
#' `0 if x < -2.5`, `1 if x > 2.5`, `(0.2 * x) + 0.5 if -2.5 <= x <= 2.5`.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-1., 0., 1.))
#' op_hard_sigmoid(x)
#' ```
#'
#' ```{r op-hard-sigmoid-plot}
#' x <- as.array(seq(-5, 5, .1))
#' plot(x, op_hard_sigmoid(x),
#'      type = 'l', panel.first = grid(), frame.plot = FALSE)
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#hardsigmoid-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/hard_sigmoid>
#' @tether keras.ops.hard_sigmoid
op_hard_sigmoid <-
function (x)
keras$ops$hard_sigmoid(x)


#' Leaky version of a Rectified Linear Unit activation function.
#'
#' @description
#' It allows a small gradient when the unit is not active, it is defined as:
#'
#' `f(x) = alpha * x for x < 0` or `f(x) = x for x >= 0`.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-1., 0., 1.))
#' op_leaky_relu(x)
#' # array([-0.2,  0. ,  1. ], shape=(3,), dtype=float64)
#' ```
#' ```{r op-leaky-relu-plot}
#' x <- seq(-5, 5, .1)
#' plot(x, op_leaky_relu(x),
#'      type = 'l', panel.first = grid())
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param negative_slope
#' Slope of the activation function at x < 0.
#' Defaults to `0.2`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#leakyrelu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/leaky_relu>
#' @tether keras.ops.leaky_relu
op_leaky_relu <-
function (x, negative_slope = 0.2)
keras$ops$leaky_relu(x, negative_slope)


#' Logarithm of the sigmoid activation function.
#'
#' @description
#' It is defined as `f(x) = log(1 / (1 + exp(-x)))`.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(-0.541391, 0.0, 0.50, 5.0))
#' op_log_sigmoid(x)
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#logsigmoid-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log_sigmoid>
#'
#' @tether keras.ops.log_sigmoid
op_log_sigmoid <-
function (x)
keras$ops$log_sigmoid(x)


#' Log-softmax activation function.
#'
#' @description
#' It is defined as:
#' `f(x) = x - max(x) - log(sum(exp(x - max(x))))`
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-1., 0., 1.))
#' op_log_softmax(x)
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Integer, axis along which the log-softmax is applied.
#' Defaults to `-1`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#logsoftmax-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log_softmax>
#'
#' @tether keras.ops.log_softmax
op_log_softmax <-
function (x, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$log_softmax, args)
}


#' Max pooling operation.
#'
#' @returns
#' A tensor of rank N+2, the result of the max pooling operation.
#'
#' @param inputs
#' Tensor of rank N+2. `inputs` has shape
#' `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
#' `data_format = "channels_last"`, or
#' `(batch_size, num_channels) + inputs_spatial_shape` if
#' `data_format = "channels_first"`. Pooling happens over the spatial
#' dimensions only.
#'
#' @param pool_size
#' int or tuple/list of integers of size
#' `len(inputs_spatial_shape)`, specifying the size of the pooling
#' window for each spatial dimension of the input tensor. If
#' `pool_size` is int, then every spatial dimension shares the same
#' `pool_size`.
#'
#' @param strides
#' int or tuple/list of integers of size
#' `len(inputs_spatial_shape)`. The stride of the sliding window for
#' each spatial dimension of the input tensor. If `strides` is int,
#' then every spatial dimension shares the same `strides`.
#'
#' @param padding
#' string, either `"valid"` or `"same"`. `"valid"` means no
#' padding is applied, and `"same"` results in padding evenly to the
#' left/right or up/down of the input such that output has the
#' same height/width dimension as the input when `strides = 1`.
#'
#' @param data_format
#' A string, either `"channels_last"` or `"channels_first"`.
#' `data_format` determines the ordering of the dimensions in the
#' inputs. If `data_format = "channels_last"`, `inputs` is of shape
#' `(batch_size, ..., channels)` while if
#' `data_format = "channels_first"`, `inputs` is of shape
#' `(batch_size, channels, ...)`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#maxpool-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/max_pool>
#' @tether keras.ops.max_pool
op_max_pool <-
function (inputs, pool_size, strides = NULL, padding = "valid",
    data_format = NULL)
{
    args <- capture_args(list(pool_size = as_integer, strides = as_integer))
    do.call(keras$ops$max_pool, args)
}


#' Calculates the mean and variance of `x`.
#'
#' @description
#' The mean and variance are calculated by aggregating the contents of `x`
#' across `axes`. If `x` is 1-D and `axes = c(1)` this is just the mean and
#' variance of a vector.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(0, 1, 2, 3, 100), dtype = "float32")
#' op_moments(x, axes = c(1))
#' ```
#'
#' @returns
#' A list containing two tensors - mean and variance.
#'
#' @param x
#' Input tensor.
#'
#' @param axes
#' A list of axes which to compute mean and variance.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduced are left
#' in the result as dimensions with size one.
#'
#' @param synchronized
#' Only applicable with the TensorFlow backend.
#' If `TRUE`, synchronizes the global batch statistics (mean and
#' variance) across all devices at each training step in a
#' distributed training strategy. If `FALSE`, each replica uses its own
#' local batch statistics.
#'
#' @export
#' @family nn ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/moments>
#'
#' @tether keras.ops.moments
op_moments <-
function (x, axes, keepdims = FALSE, synchronized = FALSE)
{
    args <- capture_args(list(axes = as_axis))
    do.call(keras$ops$moments, args)
}


#' Encodes integer labels as multi-hot vectors.
#'
#' @description
#' This function encodes integer labels as multi-hot vectors, where each label
#' is mapped to a binary value in the resulting vector.
#'
#' # Examples
#' ```{r}
#' data <- op_convert_to_tensor(c(0, 4))
#' op_multi_hot(data, num_classes = 5)
#' ```
#'
#' @returns
#' Tensor: The multi-hot encoded tensor.
#'
#' @param inputs
#' Tensor of integer labels to be converted to multi-hot vectors.
#'
#' @param num_classes
#' Integer, the total number of unique classes.
#'
#' @param axis
#' (optional) Axis along which the multi-hot encoding should be
#' added. Defaults to `-1`, which corresponds to the last dimension.
#'
#' @param dtype
#' (optional) The data type of the resulting tensor. Default
#' is backend's float type.
#'
#' @param ... For forward/backwards compatability
#'
#' @export
#' @family nn ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/multi_hot>
#'
#' @tether keras.ops.multi_hot
op_multi_hot <-
function (inputs, num_classes, axis = -1L, dtype = NULL, ...)
{
    args <- capture_args(list(inputs = as_integer, num_classes = as_integer,
        axis = as_axis))
    do.call(keras$ops$multi_hot, args)
}


#' Converts integer tensor `x` into a one-hot tensor.
#'
#' @description
#' The one-hot encoding is a representation where each integer value is
#' converted into a binary vector with a length equal to `num_classes`,
#' and the index corresponding to the integer value is marked as 1, while
#' all other indices are marked as 0.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(1, 3, 2, 0), "int32")
#' op_one_hot(x, num_classes = 4)
#' # array([[0. 1. 0. 0.]
#' #        [0. 0. 0. 1.]
#' #        [0. 0. 1. 0.]
#' #        [1. 0. 0. 0.]], shape=(4, 4), dtype=float32)
#' ```
#'
#' @returns
#' Integer tensor: One-hot encoded tensor with the same shape as `x`
#' except for the specified `axis` dimension, which will have
#' a length of `num_classes`. The dtype of the output tensor
#' is determined by `dtype` or the default data type of the backend.
#'
#' @param x
#' Integer tensor to be encoded. The shape can be
#' arbitrary, but the dtype should be integer.
#'
#' @param num_classes
#' Number of classes for the one-hot encoding.
#'
#' @param axis
#' Axis along which the encoding is performed. Defaults to
#' `-1`, which represents the last axis.
#'
#' @param dtype
#' (Optional) Data type of the output tensor. If not
#' provided, it defaults to the default data type of the backend.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#onehot-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/one_hot>
#' @tether keras.ops.one_hot
op_one_hot <-
function (x, num_classes, axis = -1L, dtype = NULL)
{
    args <- capture_args(list(x = as_integer, axis = as_axis,
        num_classes = as_integer))
    do.call(keras$ops$one_hot, args)
}


#' Rectified linear unit activation function.
#'
#' @description
#' It is defined as `f(x) = max(0, x)`.
#'
#' # Examples
#' ```{r}
#' x1 <- op_convert_to_tensor(c(-1, 0, 1, 0.2))
#' op_relu(x1)
#' ```
#'
#' ```{r op-relu-plot}
#' x <- seq(-10, 10, .1)
#' plot(x, op_relu(x))
#' ```
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#relu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/relu>
#'
#' @tether keras.ops.relu
op_relu <-
function (x)
keras$ops$relu(x)


#' Rectified linear unit activation function with upper bound of 6.
#'
#' @description
#' It is defined as `f(x) = op_clip(x, 0, 6)`.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(-3, -2, 0.1, 0.2, 6, 8))
#' op_relu6(x)
#' ```
#' ```{r op-relu6-plot}
#' x <- seq(-10, 10, .1)
#' plot(x, op_relu6(x))
#' ```
#'
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#relu6-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/relu6>
#'
#' @tether keras.ops.relu6
op_relu6 <-
function (x)
keras$ops$relu6(x)


#' Scaled Exponential Linear Unit (SELU) activation function.
#'
#' @description
#' It is defined as:
#'
#' `f(x) =  scale * alpha * (exp(x) - 1.) for x < 0`,
#' `f(x) = scale * x for x >= 0`.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-1, 0, 1))
#' op_selu(x)
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#selu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/selu>
#' @tether keras.ops.selu
op_selu <-
function (x)
keras$ops$selu(x)


#' General N-D separable convolution.
#'
#' @description
#' This ops supports 1D and 2D separable convolution. `separable_conv` is
#' a depthwise conv followed by a pointwise conv.
#'
#' @returns
#' A tensor of rank N+2, the result of the depthwise conv operation.
#'
#' @param inputs
#' Tensor of rank N+2. `inputs` has shape
#' `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
#' `data_format="channels_last"`, or
#' `(batch_size, num_channels) + inputs_spatial_shape` if
#' `data_format="channels_first"`.
#'
#' @param depthwise_kernel
#' Tensor of rank N+2. `depthwise_kernel` has shape
#' `[kernel_spatial_shape, num_input_channels, num_channels_multiplier],`
#' `num_input_channels` should match the number of channels in
#' `inputs`.
#'
#' @param pointwise_kernel
#' Tensor of rank N+2. `pointwise_kernel` has shape
#' `(*ones_like(kernel_spatial_shape),
#' num_input_channels * num_channels_multiplier, num_output_channels)`.
#'
#' @param strides
#' int or int tuple/list of `len(inputs_spatial_shape)`,
#' specifying the strides of the convolution along each spatial
#' dimension. If `strides` is int, then every spatial dimension shares
#' the same `strides`.
#'
#' @param padding
#' string, either `"valid"` or `"same"`. `"valid"` means no
#' padding is applied, and `"same"` results in padding evenly to the
#' left/right or up/down of the input such that output has the
#' same height/width dimension as the input when `strides=1`.
#'
#' @param data_format
#' A string, either `"channels_last"` or `"channels_first"`.
#' `data_format` determines the ordering of the dimensions in the
#' inputs. If `data_format="channels_last"`, `inputs` is of shape
#' `(batch_size, ..., channels)` while if
#' `data_format="channels_first"`, `inputs` is of shape
#' `(batch_size, channels, ...)`.
#'
#' @param dilation_rate
#' int or int tuple/list of `len(inputs_spatial_shape)`,
#' specifying the dilation rate to use for dilated convolution. If
#' `dilation_rate` is int, then every spatial dimension shares
#' the same `dilation_rate`.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#separableconv-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/separable_conv>
#' @tether keras.ops.separable_conv
op_separable_conv <-
function (inputs, depthwise_kernel, pointwise_kernel, strides = 1L,
    padding = "valid", data_format = NULL, dilation_rate = 1L)
{
    args <- capture_args(list(strides = as_integer, dilation_rate = as_integer))
    do.call(keras$ops$separable_conv, args)
}


#' Sigmoid activation function.
#'
#' @description
#' It is defined as `f(x) = 1 / (1 + exp(-x))`.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(-6, 1, 0, 1, 6))
#' op_sigmoid(x)
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#sigmoid-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sigmoid>
#' @tether keras.ops.sigmoid
op_sigmoid <-
function (x)
keras$ops$sigmoid(x)


#' Sigmoid Linear Unit (SiLU) activation function, also known as Swish.
#'
#' @description
#' The SiLU activation function is computed by the sigmoid function multiplied
#' by its input. It is defined as `f(x) = x * sigmoid(x)`.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(-6, 1, 0, 1, 6))
#' op_sigmoid(x)
#' op_silu(x)
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#silu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/silu>
#' @tether keras.ops.silu
op_silu <-
function (x)
keras$ops$silu(x)


#' Softmax activation function.
#'
#' @description
#' The elements of the output vector lie within the range `(0, 1)`, and their
#' total sum is exactly 1 (excluding the floating point rounding error).
#'
#' Each vector is processed independently. The `axis` argument specifies the
#' axis along which the function is applied within the input.
#'
#' It is defined as:
#' `f(x) = exp(x) / sum(exp(x))`
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(-1, 0, 1))
#' op_softmax(x)
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Integer, axis along which the softmax is applied.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#softmax-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/softmax>
#'
#' @tether keras.ops.softmax
op_softmax <-
function (x, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$softmax, args)
}


#' Softplus activation function.
#'
#' @description
#' It is defined as `f(x) = log(exp(x) + 1)`, where `log` is the natural
#' logarithm and `exp` is the exponential function.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(-0.555, 0, 0.555))
#' op_softplus(x)
#' ```
#' ```{r op-softplus-plot}
#' x <- seq(-10, 10, .1)
#' plot(x, op_softplus(x))
#' ```
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#softplus-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/softplus>
#'
#' @tether keras.ops.softplus
op_softplus <-
function (x)
keras$ops$softplus(x)


#' Softsign activation function.
#'
#' @description
#' It is defined as `f(x) = x / (abs(x) + 1)`.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(-0.100, -10.0, 1.0, 0.0, 100.0))
#' op_softsign(x)
#' ```
#' ```{r op-softsign-plot}
#' x <- seq(-10, 10, .1)
#' plot(x, op_softsign(x), ylim = c(-1, 1))
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#softsign-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/softsign>
#'
#' @tether keras.ops.softsign
op_softsign <-
function (x)
keras$ops$softsign(x)


#' Computes sparse categorical cross-entropy loss.
#'
#' @description
#' The sparse categorical cross-entropy loss is similar to categorical
#' cross-entropy, but it is used when the target tensor contains integer
#' class labels instead of one-hot encoded vectors. It measures the
#' dissimilarity between the target and output probabilities or logits.
#'
#' # Examples
#' ```{r}
#' target <- op_array(c(0, 1, 2), dtype="int32")
#' output <- op_array(rbind(c(0.9, 0.05, 0.05),
#'                         c(0.1, 0.8,  0.1),
#'                         c(0.2, 0.3,  0.5)))
#' op_sparse_categorical_crossentropy(target, output)
#' ```
#'
#' @returns
#' Integer tensor: The computed sparse categorical cross-entropy
#' loss between `target` and `output`.
#'
#' @param target
#' The target tensor representing the true class labels as
#' integers. Its shape should match the shape of the `output`
#' tensor except for the last dimension.
#'
#' @param output
#' The output tensor representing the predicted probabilities
#' or logits.
#' Its shape should match the shape of the `target` tensor except
#' for the last dimension.
#'
#' @param from_logits
#' (optional) Whether `output` is a tensor of logits
#' or probabilities.
#' Set it to `TRUE` if `output` represents logits; otherwise,
#' set it to `FALSE` if `output` represents probabilities.
#' Defaults to`FALSE`.
#'
#' @param axis
#' (optional) The axis along which the sparse categorical
#' cross-entropy is computed.
#' Defaults to `-1`, which corresponds to the last dimension
#' of the tensors.
#'
#' @export
#' @family nn ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#sparsecategoricalcrossentropy-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sparse_categorical_crossentropy>
#'
#' @tether keras.ops.sparse_categorical_crossentropy
op_sparse_categorical_crossentropy <-
function (target, output, from_logits = FALSE, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$sparse_categorical_crossentropy, args)
}


#' Compute the absolute value element-wise.
#'
#' @param x
#' Input tensor
#'
#' @returns
#' An array containing the absolute value of each element in `x`.
#'
#' @description
#'
#' # Example
#' ```{r}
#' x <- op_convert_to_tensor(c(-1.2, 1.2))
#' op_abs(x)
#' ```
#'
#' @export
#' @family numpy ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/abs>
#' @tether keras.ops.absolute
op_abs <-
function (x)
keras$ops$absolute(x)


#' Add arguments element-wise.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x1 <- op_convert_to_tensor(c(1, 4))
#' x2 <- op_convert_to_tensor(c(5, 6))
#' op_add(x1, x2)
#' # alias for x1 + x2
#' x1 + x2
#' ```
#'
#' `op_add` also broadcasts shapes:
#' ```{r}
#' x1 <- op_convert_to_tensor(array(c(5, 5, 4, 6), dim =c(2, 2)))
#' x2 <- op_convert_to_tensor(c(5, 6))
#' op_add(x1, x2)
#' ```
#'
#' @returns
#' The tensor containing the element-wise sum of `x1` and `x2`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#add-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/add>
#' @tether keras.ops.add
op_add <-
function (x1, x2)
keras$ops$add(x1, x2)


#' Test whether all array elements along a given axis evaluate to `TRUE`.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(TRUE, FALSE))
#' op_all(x)
#' ```
#'
#' ```{r}
#' (x <- op_convert_to_tensor(array(c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE), dim = c(3, 2))))
#' op_all(x, axis = 1)
#' ```
#'
#' `keepdims = TRUE` outputs a tensor with dimensions reduced to one.
#' ```{r}
#' op_all(x, keepdims = TRUE)
#' ```
#'
#' @returns
#' The tensor containing the logical AND reduction over the `axis`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' An integer or tuple of integers that represent the axis along
#' which a logical AND reduction is performed. The default
#' (`axis = NULL`) is to perform a logical AND over all the dimensions
#' of the input array. `axis` may be negative, in which case it counts
#' for the last to the first axis.
#'
#' @param keepdims
#' If `TRUE`, axes which are reduced are left in the result as
#' dimensions with size one. With this option, the result will
#' broadcast correctly against the input array. Defaults to`FALSE`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#all-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/all>
#' @tether keras.ops.all
op_all <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$all, args)
}


#  Returns the maximum of a vector or maximum value along an axis.
#
#  @description
#
#  `op_amax()` performs the same computation as [`op_max()`]
#
#  # Examples
#  ```{r, include = FALSE}
#  op_amax <- op_max
#  ```
#  ```{r}
#  (x <- op_convert_to_tensor(rbind(c(1, 3, 5), c(1, 5, 2))))
#  op_amax(x)
#  op_amax(x, axis = 1)
#  op_amax(x, axis = 1, keepdims = TRUE)
#  ```
#
#  @returns
#  A tensor with the maximum value. If `axis = NULL`, the result is a scalar
#  value representing the maximum element in the entire tensor. If `axis` is
#  given, the result is a tensor with the maximum values along
#  the specified axis.
#
#  @param x
#  Input tensor.
#
#  @param axis
#  Axis along which to compute the maximum.
#  By default (`axis = NULL`), find the maximum value in all the
#  dimensions of the input tensor.
#
#  @param keepdims
#  If `TRUE`, axes which are reduced are left in the result as
#  dimensions that are broadcast to the size of the original
#  input tensor. Defaults to `FALSE`.
#
#  @export
#  @noRd
#  @keywords internal
#  @family numpy ops
#  @family ops
#  @seealso
#  + <https://keras.io/api/ops/numpy#amax-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/amax>
#  @tether keras.ops.amax
# op_amax <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$amax, args)
}


#  Returns the minimum of a vector or minimum value along an axis.
#
#  @description
#
#  `op_amin()` performs the same computation as [`op_min()`]
#
#  # Examples
#  ```{r, include = FALSE}
#  op_amin <- op_min
#  ```
#  ```{r}
#  (x <- op_convert_to_tensor(rbind(c(1, 3, 5), c(1, 5, 2))))
#  op_amin(x)
#  op_amin(x, axis = 1)
#  op_amin(x, axis = 1, keepdims = TRUE)
#  ```
#
#  @returns
#  A tensor with the minimum value. If `axis = NULL`, the result is a scalar
#  value representing the minimum element in the entire tensor. If `axis` is
#  given, the result is a tensor with the minimum values along
#  the specified axis.
#
#  @param x
#  Input tensor.
#
#  @param axis
#  Axis along which to compute the minimum.
#  By default (`axis = NULL`), find the minimum value in all the
#  dimensions of the input tensor.
#
#  @param keepdims
#  If `TRUE`, axes which are reduced are left in the result as
#  dimensions that are broadcast to the size of the original
#  input tensor. Defaults to `FALSE`.
#
# @export
#  @noRd
#  @keywords internal
#  @family numpy ops
#  @family ops
#  @seealso
#  + <https://keras.io/api/ops/numpy#amin-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/amin>
#  @tether keras.ops.amin
# op_amin <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$amin, args)
}


#' Test whether any array element along a given axis evaluates to `TRUE`.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(TRUE, FALSE))
#' op_any(x)
#' ```
#'
#' ```{r}
#' (x <- op_reshape(c(FALSE, FALSE, FALSE,
#'                    TRUE, FALSE, FALSE),
#'                  c(2, 3)))
#' op_any(x, axis = 1)
#' op_any(x, axis = 2)
#' op_any(x, axis = -1)
#' ```
#'
#' `keepdims = TRUE` outputs a tensor with dimensions reduced to one.
#' ```{r}
#' op_any(x, keepdims = TRUE)
#' op_any(x, 1, keepdims = TRUE)
#' ```
#'
#' @returns
#' The tensor containing the logical OR reduction over the `axis`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' An integer or tuple of integers that represent the axis along
#' which a logical OR reduction is performed. The default
#' (`axis = NULL`) is to perform a logical OR over all the dimensions
#' of the input array. `axis` may be negative, in which case it counts
#' for the last to the first axis.
#'
#' @param keepdims
#' If `TRUE`, axes which are reduced are left in the result as
#' dimensions with size one. With this option, the result will
#' broadcast correctly against the input array. Defaults to `FALSE`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#any-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/any>
#' @tether keras.ops.any
op_any <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$any, args)
}


#' Append tensor `x2` to the end of tensor `x1`.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x1 <- op_convert_to_tensor(c(1, 2, 3))
#' x2 <- op_convert_to_tensor(rbind(c(4, 5, 6), c(7, 8, 9)))
#' op_append(x1, x2)
#' ```
#'
#' When `axis` is specified, `x1` and `x2` must have compatible shapes.
#' ```{r}
#' x1 <- op_convert_to_tensor(rbind(c(1, 2, 3), c(4, 5, 6)))
#' x2 <- op_convert_to_tensor(rbind(c(7, 8, 9)))
#' op_append(x1, x2, axis = 1)
#' x3 <- op_convert_to_tensor(c(7, 8, 9))
#' try(op_append(x1, x3, axis = 1))
#' ```
#'
#' @returns
#' A tensor with the values of `x2` appended to `x1`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @param axis
#' Axis along which tensor `x2` is appended to tensor `x1`.
#' If `NULL`, both tensors are flattened before use.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#append-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/append>
#' @tether keras.ops.append
op_append <-
function (x1, x2, axis = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$append, args)
}


#' Return evenly spaced values within a given interval.
#'
#' @description
#' `arange` can be called with a varying number of positional arguments:
#' * `arange(stop)`: Values are generated within the half-open interval
#'     `[0, stop)` (in other words, the interval including start but excluding
#'     stop).
#' * `arange(start, stop)`: Values are generated within the half-open interval
#'     `[start, stop)`.
#' * `arange(start, stop, step)`: Values are generated within the half-open
#'     interval `[start, stop)`, with spacing between values given by step.
#'
#' # Examples
#' ```{r}
#' op_arange(3L)
#' op_arange(3) # float
#' op_arange(3, dtype = 'int32') #int
#' op_arange(3L, 7L)
#' op_arange(3L, 7L, 2L)
#' ```
#'
#' @returns
#' Tensor of evenly spaced values.
#' For floating point arguments, the length of the result is
#' `ceiling((stop - start)/step)`. Because of floating point overflow, this
#' rule may result in the last element of out being greater than stop.
#'
#' @param start
#' Integer or real, representing the start of the interval. The
#' interval includes this value.
#'
#' @param stop
#' Integer or real, representing the end of the interval. The
#' interval does not include this value, except in some cases where
#' `step` is not an integer and floating point round-off affects the
#' length of `out`. Defaults to `NULL`.
#'
#' @param step
#' Integer or real, represent the spacing between values. For any
#' output `out`, this is the distance between two adjacent values,
#' `out[i+1] - out[i]`. The default step size is 1. If `step` is
#' specified as a position argument, `start` must also be given.
#'
#' @param dtype
#' The type of the output array. If `dtype` is not given, infer the
#' data type from the other input arguments.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#arange-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arange>
#' @tether keras.ops.arange
op_arange <-
function (start, stop = NULL, step = 1L, dtype = NULL)
{
    args <- capture_args(list(start = function (x)
    np_array(x, dtype), stop = function (x)
    np_array(x, dtype), step = function (x)
    np_array(x, dtype)))
    do.call(keras$ops$arange, args)
}


#' Trigonometric inverse cosine, element-wise.
#'
#' @description
#' The inverse of `cos` so that, if `y = cos(x)`, then `x = arccos(y)`.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(1, -1))
#' op_arccos(x)
#' ```
#'
#' @returns
#' Tensor of the angle of the ray intersecting the unit circle at the given
#' x-coordinate in radians `[0, pi]`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#arccos-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arccos>
#' @tether keras.ops.arccos
op_arccos <-
function (x)
keras$ops$arccos(x)


#' Inverse hyperbolic cosine, element-wise.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(10, 100))
#' op_arccosh(x)
#' ```
#'
#' @returns
#' Output tensor of same shape as x.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arccosh>
#' @tether keras.ops.arccosh
op_arccosh <-
function (x)
keras$ops$arccosh(x)


#' Inverse sine, element-wise.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(1, -1, 0))
#' op_arcsin(x)
#' ```
#'
#' @returns
#' Tensor of the inverse sine of each element in `x`, in radians and in
#' the closed interval `[-pi/2, pi/2]`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#arcsin-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arcsin>
#' @tether keras.ops.arcsin
op_arcsin <-
function (x)
keras$ops$arcsin(x)


#' Inverse hyperbolic sine, element-wise.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(1, -1, 0))
#' op_arcsinh(x)
#' ```
#'
#' @returns
#' Output tensor of same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#arcsinh-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arcsinh>
#' @tether keras.ops.arcsinh
op_arcsinh <-
function (x)
keras$ops$arcsinh(x)


#' Trigonometric inverse tangent, element-wise.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(0, 1))
#' op_arctan(x)
#' ```
#'
#' @returns
#' Tensor of the inverse tangent of each element in `x`, in the interval
#' `[-pi/2, pi/2]`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#arctan-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arctan>
#' @tether keras.ops.arctan
op_arctan <-
function (x)
keras$ops$arctan(x)


#' Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.
#'
#' @description
#' The quadrant (i.e., branch) is chosen so that `arctan2(x1, x2)` is the
#' signed angle in radians between the ray ending at the origin and passing
#' through the point `(1, 0)`, and the ray ending at the origin and passing
#' through the point `(x2, x1)`. (Note the role reversal: the "y-coordinate"
#' is the first function parameter, the "x-coordinate" is the second.) By IEEE
#' convention, this function is defined for `x2 = +/-0` and for either or both
#' of `x1` and `x2` `= +/-inf`.
#'
#' # Examples
#' Consider four points in different quadrants:
#' ```{r}
#' x <- op_array(c(-1, 1, 1, -1))
#' y <- op_array(c(-1, -1, 1, 1))
#' op_arctan2(y, x) * 180 / pi
#' ```
#'
#' Note the order of the parameters. `arctan2` is defined also when x2 = 0 and
#' at several other points, obtaining values in the range `[-pi, pi]`:
#' ```{r}
#' op_arctan2(
#'     op_array(c(1, -1)),
#'     op_array(c(0, 0))
#' )
#' op_arctan2(
#'     op_array(c(0, 0, Inf)),
#'     op_array(c(+0, -0, Inf))
#' )
#' ```
#'
#' @returns
#' Tensor of angles in radians, in the range `[-pi, pi]`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#arctan2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arctan2>
#' @tether keras.ops.arctan2
op_arctan2 <-
function (x1, x2)
keras$ops$arctan2(x1, x2)


#' Inverse hyperbolic tangent, element-wise.
#'
#' @returns
#' Output tensor of same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#arctanh-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arctanh>
#' @tether keras.ops.arctanh
op_arctanh <-
function (x)
keras$ops$arctanh(x)


#' Returns the indices of the maximum values along an axis.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_arange(6L) |> op_reshape(c(2, 3)) |> op_add(10)
#' x
#' op_argmax(x)
#' op_argmax(x, axis = 1)
#' op_argmax(x, axis = 2)
#' ```
#'
#' @returns
#' Tensor of indices. It has the same shape as `x`, with the dimension
#' along `axis` removed. Note that the returned integer is 0-based (i.e., if the
#' argmax is in the first index position, the returned value will be `0`)
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' By default, the index is into the flattened tensor, otherwise
#' along the specified axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#argmax-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/argmax>
#' @tether keras.ops.argmax
op_argmax <-
function (x, axis = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$argmax, args)
}


#' Returns the indices of the minimum values along an axis.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_arange(6L) |> op_reshape(c(2, 3)) |> op_add(10)
#' x
#' op_argmin(x)
#' op_argmin(x, axis = 1)
#' op_argmin(x, axis = 2)
#' ```
#'
#' @returns
#' Tensor of indices. It has the same shape as `x`, with the dimension
#' along `axis` removed.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' By default, the index is into the flattened tensor, otherwise
#' along the specified axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#argmin-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/argmin>
#' @tether keras.ops.argmin
op_argmin <-
function (x, axis = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$argmin, args)
}


#' Returns the indices that would sort a tensor.
#'
#' @description
#'
#' # Examples
#' One dimensional array:
#' ```{r}
#' x <- op_array(c(3, 1, 2))
#' op_argsort(x)
#' ```
#'
#' Two-dimensional array:
#' ```{r}
#' x <- op_array(rbind(c(0, 3),
#'                    c(3, 2),
#'                    c(4, 5)), dtype = "int32")
#' op_argsort(x, axis = 1)
#' op_argsort(x, axis = 2)
#' ```
#'
#' @returns
#' Tensor of indices that sort `x` along the specified `axis`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis along which to sort. Defaults to `-1` (the last axis). If
#' `NULL`, the flattened tensor is used.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#argsort-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/argsort>
#' @tether keras.ops.argsort
op_argsort <-
function (x, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$argsort, args)
}


#' Create a tensor.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' op_array(c(1, 2, 3))
#' op_array(c(1, 2, 3), dtype = "float32")
#' ```
#'
#' @returns
#' A tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param dtype
#' The desired data-type for the tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#array-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/array>
#' @tether keras.ops.array
op_array <-
function (x, dtype = NULL)
{
    if (!is.null(dtype) && !inherits(x, "python.builtin.object"))
        x <- np_array(x, dtype)
    keras$ops$array(x, dtype)
}


#' Compute the weighted average along the specified axis.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' data <- op_arange(1, 5, dtype = "int32")
#' data
#' op_average(data)
#'
#' op_average(
#'   op_arange(1, 11),
#'   weights = op_arange(10, 0, -1)
#' )
#'
#' data <- op_arange(6) |> op_reshape(c(3, 2))
#' data
#'
#' op_average(
#'   data,
#'   axis = 2,
#'   weights = op_array(c(1/4, 3/4))
#' )
#' # Error: Axis must be specified when shapes of a and weights differ.
#' try(op_average(
#'   data,
#'   weights = op_array(c(1/4, 3/4))
#' ))
#' ```
#'
#' @returns
#' Return the average along the specified axis.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Integer along which to average `x`. The default, `axis = NULL`,
#' will average over all of the elements of the input tensor. If axis
#' is negative it counts from the last to the first axis.
#'
#' @param weights
#' Tensor of wieghts associated with the values in `x`. Each
#' value in `x` contributes to the average according to its
#' associated weight. The weights array can either be 1-D (in which
#' case its length must be the size of a along the given axis) or of
#' the same shape as `x`. If `weights = NULL` (default), then all data
#' in `x` are assumed to have a weight equal to one.
#'
#' The 1-D calculation is: `avg = sum(a * weights) / sum(weights)`.
#' The only constraint on weights is that `sum(weights)` must not be 0.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#average-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/average>
#' @tether keras.ops.average
op_average <-
function (x, axis = NULL, weights = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$average, args)
}


#' Count the number of occurrences of each value in a tensor of integers.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' (x <- op_array(c(1, 2, 2, 3), dtype = "uint8"))
#' op_bincount(x)
#'
#' (weights <- x / 2)
#' op_bincount(x, weights = weights)
#'
#' minlength <- as.integer(op_max(x) + 1 + 2) # 6
#' op_bincount(x, minlength = minlength)
#' ```
#'
#' @returns
#' 1D tensor where each element gives the number of occurrence(s) of its
#' index value in x. Its length is the maximum between `max(x) + 1` and
#' minlength.
#'
#' @param x
#' Input tensor.
#' It must be of dimension 1, and it must only contain non-negative
#' integer(s).
#'
#' @param weights
#' Weight tensor.
#' It must have the same length as `x`. The default value is `NULL`.
#' If specified, `x` is weighted by it, i.e. if `n = x[i]`,
#' `out[n] += weight[i]` instead of the default behavior `out[n] += 1`.
#'
#' @param minlength
#' An integer.
#' The default value is 0. If specified, there will be at least
#' this number of bins in the output tensor. If greater than
#' `max(x) + 1`, each value of the output at an index higher than
#' `max(x)` is set to 0.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#bincount-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/bincount>
#' @tether keras.ops.bincount
op_bincount <-
function (x, weights = NULL, minlength = 0L)
{
    args <- capture_args(list(x = as_integer, minlength = as_integer))
    do.call(keras$ops$bincount, args)
}


#' Broadcast a tensor to a new shape.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(1, 2, 3))
#' op_broadcast_to(x, shape = c(3, 3))
#' ```
#'
#' @returns
#' A tensor with the desired shape.
#'
#' @param x
#' The tensor to broadcast.
#'
#' @param shape
#' The shape of the desired tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#broadcastto-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/broadcast_to>
#' @tether keras.ops.broadcast_to
op_broadcast_to <-
function (x, shape)
{
    args <- capture_args(list(shape = normalize_shape))
    do.call(keras$ops$broadcast_to, args)
}


#' Return the ceiling of the input, element-wise.
#'
#' @description
#' The ceil of the scalar `x` is the smallest integer `i`, such that
#' `i >= x`.
#'
#' @returns
#' The ceiling of each element in `x`, with float dtype.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#ceil-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/ceil>
#' @tether keras.ops.ceil
op_ceil <-
function (x)
keras$ops$ceil(x)


#' Clip (limit) the values in a tensor.
#'
#' @description
#' Given an interval, values outside the interval are clipped to the
#' interval edges. For example, if an interval of `[0, 1]` is specified,
#' values smaller than 0 become 0, and values larger than 1 become 1.
#'
#' @returns
#' The clipped tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param x_min
#' Minimum value.
#'
#' @param x_max
#' Maximum value.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#clip-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/clip>
#' @tether keras.ops.clip
op_clip <-
function (x, x_min, x_max)
keras$ops$clip(x, x_min, x_max)


#' Join a sequence of tensors along an existing axis.
#'
#' @returns
#' The concatenated tensor.
#'
#' @param xs
#' The sequence of tensors to concatenate.
#'
#' @param axis
#' The axis along which the tensors will be joined. Defaults to `0`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#concatenate-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/concatenate>
#' @tether keras.ops.concatenate
op_concatenate <-
function (xs, axis = 1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$concatenate, args)
}


# ' Shorthand for [`op_conjugate()`].
# '
# ' @param x
# ' see description
# '
# ' @export
# ' @family numpy ops
# ' @family ops
# ' @seealso
# ' + <https://keras.io/api/ops/numpy#conj-function>
# ' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/conj>
# ' @tether keras.ops.conj
# op_conj <-
# function (x)
# keras$ops$conj(x)


#' Returns the complex conjugate, element-wise.
#'
#' @description
#' The complex conjugate of a complex number is obtained by changing the sign
#' of its imaginary part.
#'
#' @returns
#' The complex conjugate of each element in `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#conjugate-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/conjugate>
#' @tether keras.ops.conjugate
op_conj <-
function (x)
keras$ops$conjugate(x)


#' Returns a copy of `x`.
#'
#' @returns
#' A copy of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#copy-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/copy>
#' @tether keras.ops.copy
op_copy <-
function (x)
keras$ops$copy(x)


#' Cosine, element-wise.
#'
#' @returns
#' The corresponding cosine values.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#cos-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cos>
#' @tether keras.ops.cos
op_cos <-
function (x)
keras$ops$cos(x)


#' Hyperbolic cosine, element-wise.
#'
#' @returns
#' Output tensor of same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#cosh-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cosh>
#' @tether keras.ops.cosh
op_cosh <-
function (x)
keras$ops$cosh(x)


#' Counts the number of non-zero values in `x` along the given `axis`.
#'
#' @description
#' If no axis is specified then all non-zeros in the tensor are counted.
#'
#' # Examples
#' ```{r}
#' x <- op_array(rbind(c(0, 1, 7, 0),
#'                    c(3, 0, 2, 19)))
#' op_count_nonzero(x)
#' op_count_nonzero(x, axis = 1)
#'
#' op_count_nonzero(x, axis = 2)
#' ```
#'
#' @returns
#' An integer or a tensor of integers.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or a tuple of axes along which to count the number of
#' non-zeros. Defaults to `NULL`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#countnonzero-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/count_nonzero>
#' @tether keras.ops.count_nonzero
op_count_nonzero <-
function (x, axis = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$count_nonzero, args)
}


#' Returns the cross product of two (arrays of) vectors.
#'
#' @description
#' The cross product of `x1` and `x2` in R^3 is a vector
#' perpendicular to both `x1` and `x2`. If `x1` and `x2` are arrays of
#' vectors, the vectors are defined by the last axis of `x1` and `x2`
#' by default, and these axes can have dimensions 2 or 3.
#'
#' Where the dimension of either `x1` or `x2` is 2, the third component of
#' the input vector is assumed to be zero and the cross product calculated
#' accordingly.
#'
#' In cases where both input vectors have dimension 2, the z-component of
#' the cross product is returned.
#'
#' # Note
#' Torch backend does not support two dimensional vectors, or the
#' arguments `axisa`, `axisb` and `axisc`. Use `axis` instead.
#'
#' @returns
#' Vector cross product(s).
#'
#' @param x1
#' Components of the first vector(s).
#'
#' @param x2
#' Components of the second vector(s).
#'
#' @param axisa
#' Axis of `x1` that defines the vector(s). Defaults to `-1`.
#'
#' @param axisb
#' Axis of `x2` that defines the vector(s). Defaults to `-1`.
#'
#' @param axisc
#' Axis of the result containing the cross product vector(s).
#' Ignored if both input vectors have dimension 2, as the return is
#' scalar. By default, the last axis.
#'
#' @param axis
#' If defined, the axis of `x1`, `x2` and the result that
#' defines the vector(s) and cross product(s). Overrides `axisa`,
#' `axisb` and `axisc`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#cross-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cross>
#' @tether keras.ops.cross
op_cross <-
function (x1, x2, axisa = -1L, axisb = -1L, axisc = -1L, axis = NULL)
{
    args <- capture_args(list(axisa = as_integer, axisb = as_integer,
        axisc = as_integer, axis = as_axis))
    do.call(keras$ops$cross, args)
}


#' Return the cumulative product of elements along a given axis.
#'
#' @returns
#' Output tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis along which the cumulative product is computed.
#' By default the input is flattened.
#'
#' @param dtype
#' dtype of returned tensor. Defaults to `x$dtype`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#cumprod-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cumprod>
#' @tether keras.ops.cumprod
op_cumprod <-
function (x, axis = NULL, dtype = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$cumprod, args)
}


#' Returns the cumulative sum of elements along a given axis.
#'
#' @returns
#' Output tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis along which the cumulative sum is computed.
#' By default the input is flattened.
#'
#' @param dtype
#' dtype of returned tensor. Defaults to `x$dtype`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#cumsum-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cumsum>
#' @tether keras.ops.cumsum
op_cumsum <-
function (x, axis = NULL, dtype = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$cumsum, args)
}


#' Extract a diagonal or construct a diagonal array.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_arange(9L) |> op_reshape(c(3, 3))
#' x
#' op_diag(x)
#' op_diag(x, k = 1)
#' op_diag(x, k = -1)
#' op_diag(op_diag(x))
#' ```
#'
#' @returns
#' The extracted diagonal or constructed diagonal tensor.
#'
#' @param x
#' Input tensor. If `x` is 2-D, returns the k-th diagonal of `x`.
#' If `x` is 1-D, return a 2-D tensor with `x` on the k-th diagonal.
#'
#' @param k
#' The diagonal to consider. Defaults to `0`. Use `k > 0` for diagonals
#' above the main diagonal, and `k < 0` for diagonals below
#' the main diagonal.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#diag-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/diag>
#' @tether keras.ops.diag
op_diag <-
function (x, k = 0L)
{
    args <- capture_args(list(k = as_integer))
    do.call(keras$ops$diag, args)
}


#' Return specified diagonals.
#'
#' @description
#' If `x` is 2-D, returns the diagonal of `x` with the given offset, i.e., the
#' collection of elements of the form `x[i, i+offset]`.
#'
#' If `x` has more than two dimensions, the axes specified by `axis1`
#' and `axis2` are used to determine the 2-D sub-array whose diagonal
#' is returned.
#'
#' The shape of the resulting array can be determined by removing `axis1`
#' and `axis2` and appending an index to the right equal to the size of
#' the resulting diagonals.
#'
#' # Examples
#' ```{r}
#' x <- op_arange(4L) |> op_reshape(c(2, 2))
#' x
#' op_diagonal(x)
#' op_diagonal(x, offset = 1)
#'
#' x <- op_array(1:8) |> op_reshape(c(2, 2, 2))
#' x
#' x |> op_diagonal(0)
#' x |> op_diagonal(0, 1, 2) # same as above, the default
#' x |> op_diagonal(0, 2, 3)
#' ```
#'
#' @returns
#' Tensor of diagonals.
#'
#' @param x
#' Input tensor.
#'
#' @param offset
#' Offset of the diagonal from the main diagonal.
#' Can be positive or negative. Defaults to `0` (main diagonal).
#'
#' @param axis1
#' Axis to be used as the first axis of the 2-D sub-arrays.
#' Defaults to `1` (first axis).
#'
#' @param axis2
#' Axis to be used as the second axis of the 2-D sub-arrays.
#' Defaults to `2` (second axis).
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#diagonal-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/diagonal>
#' @tether keras.ops.diagonal
op_diagonal <-
function (x, offset = 0L, axis1 = 1L, axis2 = 2L)
{
    args <- capture_args(list(offset = as_integer, axis1 = as_axis,
        axis2 = as_axis))
    do.call(keras$ops$diagonal, args)
}


#' Calculate the n-th discrete difference along the given axis.
#'
#' @description
#' The first difference is given by `out[i] = a[i+1] - a[i]` along
#' the given axis, higher differences are calculated by using `diff`
#' recursively.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(1, 2, 4, 7, 0))
#' op_diff(x)
#' op_diff(x, n = 2)
#' x <- op_array(rbind(c(1, 3, 6, 10),
#'                   c(0, 5, 6, 8)))
#' op_diff(x)
#' op_diff(x, axis = 1)
#' ```
#'
#' @returns
#' Tensor of diagonals.
#'
#' @param a
#' Input tensor.
#'
#' @param n
#' The number of times values are differenced. Defaults to `1`.
#'
#' @param axis
#' Axis to compute discrete difference(s) along.
#' Defaults to `-1` (last axis).
#'
#' @export
#' @family numpy ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/diff>
#' @tether keras.ops.diff
op_diff <-
function (a, n = 1L, axis = -1L)
{
    args <- capture_args(list(n = as_integer, axis = as_axis))
    do.call(keras$ops$diff, args)
}


#' Returns the indices of the bins to which each value in `x` belongs.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(0.0, 1.0, 3.0, 1.6))
#' bins <- array(c(0.0, 3.0, 4.5, 7.0))
#' op_digitize(x, bins)
#' # array([1, 1, 2, 1])
#' ```
#'
#' @returns
#' Output array of indices, of same shape as `x`.
#'
#' @param x
#' Input array to be binned.
#'
#' @param bins
#' Array of bins. It has to be one-dimensional and monotonically
#' increasing.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#digitize-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/digitize>
#' @tether keras.ops.digitize
op_digitize <-
function (x, bins)
{
    args <- capture_args(list(bins = as.array))
    do.call(keras$ops$digitize, args)
}


#' Divide arguments element-wise.
#'
#' @returns
#' Output tensor, the quotient `x1/x2`, element-wise.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @details
#'
#' # Example
#' ```{r}
#' op_divide(3, 2)
#' ```
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#divide-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/divide>
#' @tether keras.ops.divide
op_divide <-
function (x1, x2)
keras$ops$divide(x1, x2)


#' Dot product of two tensors.
#'
#' @description
#' - If both `x1` and `x2` are 1-D tensors, it is inner product of vectors
#'   (without complex conjugation).
#' - If both `x1` and `x2` are 2-D tensors, it is matrix multiplication.
#' - If either `x1` or `x2` is 0-D (scalar), it is equivalent to `x1 * x2`.
#' - If `x1` is an N-D tensor and `x2` is a 1-D tensor, it is a sum product
#'   over the last axis of `x1` and `x2`.
#' - If `x1` is an N-D tensor and `x2` is an M-D tensor (where `M >= 2`),
#'   it is a sum product over the last axis of `x1` and the second-to-last
#'   axis of `x2`: `dot(x1, x2)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`.
#'
#' # Note
#' Torch backend does not accept 0-D tensors as arguments.
#'
#' @returns
#' Dot product of `x1` and `x2`.
#'
#' @param x1
#' First argument.
#'
#' @param x2
#' Second argument.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#dot-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/dot>
#' @tether keras.ops.dot
op_dot <-
function (x1, x2)
keras$ops$dot(x1, x2)


#' Evaluates the Einstein summation convention on the operands.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' a <- op_arange(25) |> op_reshape(c(5, 5))
#' b <- op_arange(5)
#' c <- op_arange(6) |> op_reshape(c(2, 3))
#' ```
#'
#' Trace of a matrix:
#'
#' ```{r, results = 'hold'}
#' op_einsum("ii", a)
#' op_trace(a)
#' ```
#'
#' Extract the diagonal:
#'
#' ```{r, results = 'hold'}
#' op_einsum("ii -> i", a)
#' op_diag(a)
#' ```
#'
#' Sum over an axis:
#'
#' ```{r, results = 'hold'}
#' op_einsum("ij -> i", a)
#' op_sum(a, axis = 2)
#' ```
#'
#' For higher dimensional tensors summing a single axis can be done
#' with ellipsis:
#'
#' ```{r, results = 'hold'}
#' op_einsum("...j -> ...", a)
#' op_sum(a, axis = -1)
#' ```
#'
#' Compute a matrix transpose or reorder any number of axes:
#'
#' ```{r, results = 'hold'}
#' op_einsum("ji", c)
#' op_einsum("ij -> ji", c)
#' op_transpose(c)
#' ```
#'
#' Matrix vector multiplication:
#'
#' ```{r, results = 'hold'}
#' op_einsum("ij, j", a, b)
#' op_einsum("...j, j", a, b)
#' a %*% b
#' op_matmul(a, b)
#' ```
#'
#' @returns
#' The calculation based on the Einstein summation convention.
#'
#' @param subscripts
#' Specifies the subscripts for summation as comma separated
#' list of subscript labels. An implicit (classical Einstein
#' summation) calculation is performed unless the explicit indicator
#' `->` is included as well as subscript labels of the precise
#' output form.
#'
#' @param ...
#' The operands to compute the Einstein sum of.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#einsum-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/einsum>
#' @tether keras.ops.einsum
op_einsum <-
function (subscripts, ...)
keras$ops$einsum(subscripts, ...)


#' Return a tensor of given shape and type filled with uninitialized data.
#'
#' @returns
#' The empty tensor.
#'
#' @param shape
#' Shape of the empty tensor.
#'
#' @param dtype
#' Desired data type of the empty tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#empty-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/empty>
#' @tether keras.ops.empty
op_empty <-
function (shape, dtype = NULL)
{
    args <- capture_args(list(shape = normalize_shape))
    do.call(keras$ops$empty, args)
}


#' Returns `(x1 == x2)` element-wise.
#'
#' @returns
#' Output tensor, element-wise comparison of `x1` and `x2`.
#'
#' @param x1
#' Tensor to compare.
#'
#' @param x2
#' Tensor to compare.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#equal-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/equal>
#' @tether keras.ops.equal
op_equal <-
function (x1, x2)
keras$ops$equal(x1, x2)


#' Calculate the exponential of all elements in the input tensor.
#'
#' @returns
#' Output tensor, element-wise exponential of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#exp-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/exp>
#' @tether keras.ops.exp
op_exp <-
function (x)
keras$ops$exp(x)


#' Expand the shape of a tensor.
#'
#' @description
#' Insert a new axis at the `axis` position in the expanded tensor shape.
#'
#' @returns
#' Output tensor with the number of dimensions increased.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Position in the expanded axes where the new axis
#' (or axes) is placed.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#expanddims-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/expand_dims>
#' @tether keras.ops.expand_dims
op_expand_dims <-
function (x, axis)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$expand_dims, args)
}


#' Calculate `exp(x) - 1` for all elements in the tensor.
#'
#' @returns
#' Output tensor, element-wise exponential minus one.
#'
#' @param x
#' Input values.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#expm1-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/expm1>
#' @tether keras.ops.expm1
op_expm1 <-
function (x)
keras$ops$expm1(x)


#' Return a 2-D tensor with ones on the diagonal and zeros elsewhere.
#'
#' @returns
#' Tensor with ones on the k-th diagonal and zeros elsewhere.
#'
#' @param N
#' Number of rows in the output.
#'
#' @param M
#' Number of columns in the output. If `NULL`, defaults to `N`.
#'
#' @param k
#' Index of the diagonal: 0 (the default) refers to the main
#' diagonal, a positive value refers to an upper diagonal,
#' and a negative value to a lower diagonal.
#'
#' @param dtype
#' Data type of the returned tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#eye-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/eye>
#' @tether keras.ops.eye
op_eye <-
function (N, M = NULL, k = 0L, dtype = NULL)
{
    args <- capture_args(list(k = as_integer))
    do.call(keras$ops$eye, args)
}


#' Reverse the order of elements in the tensor along the given axis.
#'
#' @description
#' The shape of the tensor is preserved, but the elements are reordered.
#'
#' @returns
#' Output tensor with entries of `axis` reversed.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which to flip the tensor. The default,
#' `axis = NULL`, will flip over all of the axes of the input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#flip-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/flip>
#' @tether keras.ops.flip
op_flip <-
function (x, axis = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$flip, args)
}


#' Return the floor of the input, element-wise.
#'
#' @description
#' The floor of the scalar `x` is the largest integer `i`, such that `i <= x`.
#'
#' @returns
#' Output tensor, element-wise floor of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#floor-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/floor>
#' @tether keras.ops.floor
op_floor <-
function (x)
keras$ops$floor(x)


#' Returns the largest integer smaller or equal to the division of inputs.
#'
#' @returns
#' Output tensor, `y <- floor(x1/x2)`
#'
#' @param x1
#' Numerator.
#'
#' @param x2
#' Denominator.
#'
#' @export
#' @family numpy ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/floor_divide>
#' @tether keras.ops.floor_divide
op_floor_divide <-
function (x1, x2)
keras$ops$floor_divide(x1, x2)


#' Return a new tensor of given shape and type, filled with `fill_value`.
#'
#' @returns
#' Output tensor.
#'
#' @param shape
#' Shape of the new tensor.
#'
#' @param fill_value
#' Fill value.
#'
#' @param dtype
#' Desired data type of the tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#full-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/full>
#' @tether keras.ops.full
op_full <-
function (shape, fill_value, dtype = NULL)
{
    args <- capture_args(list(shape = normalize_shape))
    do.call(keras$ops$full, args)
}


#' Return a full tensor with the same shape and type as the given tensor.
#'
#' @returns
#' Tensor of `fill_value` with the same shape and type as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param fill_value
#' Fill value.
#'
#' @param dtype
#' Overrides data type of the result.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#fulllike-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/full_like>
#' @tether keras.ops.full_like
op_full_like <-
function (x, fill_value, dtype = NULL)
keras$ops$full_like(x, fill_value, dtype)


#' Return `x[key]`.
#'
#' @param x
#' A dictionary-like object
#'
#' @param key
#' Generally, a string, but most object with a `__hash__` method are acceptable.
#'
#' @note
#' Generally, calling `x[[key]]` or `x$key` is preferable.
#'
#' @returns `key`.
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#getitem-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/get_item>
#' @tether keras.ops.get_item
op_get_item <-
function (x, key)
keras$ops$get_item(x, key)


#' Return the truth value of `x1 > x2` element-wise.
#'
#' @returns
#' Output tensor, element-wise comparison of `x1` and `x2`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#greater-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/greater>
#' @tether keras.ops.greater
op_greater <-
function (x1, x2)
keras$ops$greater(x1, x2)


#' Return the truth value of `x1 >= x2` element-wise.
#'
#' @returns
#' Output tensor, element-wise comparison of `x1` and `x2`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#greaterequal-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/greater_equal>
#' @tether keras.ops.greater_equal
op_greater_equal <-
function (x1, x2)
keras$ops$greater_equal(x1, x2)


#' Stack tensors in sequence horizontally (column wise).
#'
#' @description
#' This is equivalent to concatenation along the first axis for 1-D tensors,
#' and along the second axis for all other tensors.
#'
#' @returns
#' The tensor formed by stacking the given tensors.
#'
#' @param xs
#' Sequence of tensors.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#hstack-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/hstack>
#' @tether keras.ops.hstack
op_hstack <-
function (xs)
keras$ops$hstack(xs)


#' Return the identity tensor.
#'
#' @description
#' The identity tensor is a square tensor with ones on the main diagonal and
#' zeros elsewhere.
#'
#' @returns
#' The identity tensor.
#'
#' @param n
#' Number of rows (and columns) in the `n x n` output tensor.
#'
#' @param dtype
#' Data type of the output tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#identity-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/identity>
#' @tether keras.ops.identity
op_identity <-
function (n, dtype = NULL)
keras$ops$identity(n, dtype)


#' Return the imaginary part of the complex argument.
#'
#' @returns
#' The imaginary component of the complex argument.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#imag-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/imag>
#' @tether keras.ops.imag
op_imag <-
function (x)
keras$ops$imag(x)


#' Return whether two tensors are element-wise almost equal.
#'
#' @returns
#' Output boolean tensor.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#isclose-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/isclose>
#' @tether keras.ops.isclose
op_isclose <-
function (x1, x2)
keras$ops$isclose(x1, x2)


#' Return whether a tensor is finite, element-wise.
#'
#' @description
#' Real values are finite when they are not NaN, not positive infinity, and
#' not negative infinity. Complex values are finite when both their real
#' and imaginary parts are finite.
#'
#' @returns
#' Output boolean tensor.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#isfinite-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/isfinite>
#' @tether keras.ops.isfinite
op_isfinite <-
function (x)
keras$ops$isfinite(x)


#' Test element-wise for positive or negative infinity.
#'
#' @returns
#' Output boolean tensor.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#isinf-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/isinf>
#' @tether keras.ops.isinf
op_isinf <-
function (x)
keras$ops$isinf(x)


#' Test element-wise for NaN and return result as a boolean tensor.
#'
#' @returns
#' Output boolean tensor.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#isnan-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/isnan>
#' @tether keras.ops.isnan
op_isnan <-
function (x)
keras$ops$isnan(x)


#' Return the truth value of `x1 < x2` element-wise.
#'
#' @returns
#' Output tensor, element-wise comparison of `x1` and `x2`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#less-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/less>
#' @tether keras.ops.less
op_less <-
function (x1, x2)
keras$ops$less(x1, x2)


#' Return the truth value of `x1 <= x2` element-wise.
#'
#' @returns
#' Output tensor, element-wise comparison of `x1` and `x2`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#lessequal-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/less_equal>
#' @tether keras.ops.less_equal
op_less_equal <-
function (x1, x2)
keras$ops$less_equal(x1, x2)


#' Return evenly spaced numbers over a specified interval.
#'
#' @description
#' Returns `num` evenly spaced samples, calculated over the interval
#' `[start, stop]`.
#'
#' The endpoint of the interval can optionally be excluded.
#'
#' # Note
#' Torch backend does not support `axis` argument.
#'
#' @returns
#' A tensor of evenly spaced numbers.
#' If `retstep` is `TRUE`, returns `(samples, step)`
#'
#' @param start
#' The starting value of the sequence.
#'
#' @param stop
#' The end value of the sequence, unless `endpoint` is set to
#' `FALSE`. In that case, the sequence consists of all but the last
#' of `num + 1` evenly spaced samples, so that `stop` is excluded.
#' Note that the step size changes when `endpoint` is `FALSE`.
#'
#' @param num
#' Number of samples to generate. Defaults to `50`. Must be
#' non-negative.
#'
#' @param endpoint
#' If `TRUE`, `stop` is the last sample. Otherwise, it is
#' not included. Defaults to`TRUE`.
#'
#' @param retstep
#' If `TRUE`, return `(samples, step)`, where `step` is the
#' spacing between samples.
#'
#' @param dtype
#' The type of the output tensor.
#'
#' @param axis
#' The axis in the result to store the samples. Relevant only if
#' start or stop are array-like. Defaults to `1`, the first axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#linspace-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/linspace>
#' @tether keras.ops.linspace
op_linspace <-
function (start, stop, num = 50L, endpoint = TRUE, retstep = FALSE,
    dtype = NULL, axis = 1L)
{
    args <- capture_args(list(num = as_integer, axis = as_axis))
    do.call(keras$ops$linspace, args)
}


#' Natural logarithm, element-wise.
#'
#' @returns
#' Output tensor, element-wise natural logarithm of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#log-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log>
#' @tether keras.ops.log
op_log <-
function (x)
keras$ops$log(x)


#' Return the base 10 logarithm of the input tensor, element-wise.
#'
#' @returns
#' Output tensor, element-wise base 10 logarithm of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#log10-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log10>
#' @tether keras.ops.log10
op_log10 <-
function (x)
keras$ops$log10(x)


#' Returns the natural logarithm of one plus the `x`, element-wise.
#'
#' @description
#' Calculates `log(1 + x)`.
#'
#' @returns
#' Output tensor, element-wise natural logarithm of `1 + x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#log1p-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log1p>
#' @tether keras.ops.log1p
op_log1p <-
function (x)
keras$ops$log1p(x)


#' Base-2 logarithm of `x`, element-wise.
#'
#' @returns
#' Output tensor, element-wise base-2 logarithm of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#log2-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log2>
#' @tether keras.ops.log2
op_log2 <-
function (x)
keras$ops$log2(x)


#' Logarithm of the sum of exponentiations of the inputs.
#'
#' @description
#' Calculates `log(exp(x1) + exp(x2))`.
#'
#' @returns
#' Output tensor, element-wise logarithm of the sum of exponentiations
#' of the inputs.
#'
#' @param x1
#' Input tensor.
#'
#' @param x2
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#logaddexp-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logaddexp>
#' @tether keras.ops.logaddexp
op_logaddexp <-
function (x1, x2)
keras$ops$logaddexp(x1, x2)


#' Computes the element-wise logical AND of the given input tensors.
#'
#' @description
#' Zeros are treated as `FALSE` and non-zeros are treated as `TRUE`.
#'
#' @returns
#' Output tensor, element-wise logical AND of the inputs.
#'
#' @param x1
#' Input tensor.
#'
#' @param x2
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#logicaland-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logical_and>
#' @tether keras.ops.logical_and
op_logical_and <-
function (x1, x2)
keras$ops$logical_and(x1, x2)


#' Computes the element-wise NOT of the given input tensor.
#'
#' @description
#' Zeros are treated as `FALSE` and non-zeros are treated as `TRUE`.
#'
#' @returns
#' Output tensor, element-wise logical NOT of the input.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#logicalnot-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logical_not>
#' @tether keras.ops.logical_not
op_logical_not <-
function (x)
keras$ops$logical_not(x)


#' Computes the element-wise logical OR of the given input tensors.
#'
#' @description
#' Zeros are treated as `FALSE` and non-zeros are treated as `TRUE`.
#'
#' @returns
#' Output tensor, element-wise logical OR of the inputs.
#'
#' @param x1
#' Input tensor.
#'
#' @param x2
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#logicalor-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logical_or>
#' @tether keras.ops.logical_or
op_logical_or <-
function (x1, x2)
keras$ops$logical_or(x1, x2)


#' Compute the truth value of `x1 XOR x2`, element-wise.
#'
#' @returns
#' Output boolean tensor.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logical_xor>
#' @tether keras.ops.logical_xor
op_logical_xor <-
function (x1, x2)
keras$ops$logical_xor(x1, x2)


#' Returns numbers spaced evenly on a log scale.
#'
#' @description
#' In linear space, the sequence starts at `base ** start` and ends with
#' `base ** stop` (see `endpoint` below).
#'
#' # Note
#' Torch backend does not support `axis` argument.
#'
#' @returns
#' A tensor of evenly spaced samples on a log scale.
#'
#' @param start
#' The starting value of the sequence.
#'
#' @param stop
#' The final value of the sequence, unless `endpoint` is `FALSE`.
#' In that case, `num + 1` values are spaced over the interval in
#' log-space, of which all but the last (a sequence of length `num`)
#' are returned.
#'
#' @param num
#' Number of samples to generate. Defaults to `50`.
#'
#' @param endpoint
#' If `TRUE`, `stop` is the last sample. Otherwise, it is not
#' included. Defaults to`TRUE`.
#'
#' @param base
#' The base of the log space. Defaults to `10`.
#'
#' @param dtype
#' The type of the output tensor.
#'
#' @param axis
#' The axis in the result to store the samples. Relevant only
#' if start or stop are array-like.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#logspace-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logspace>
#' @tether keras.ops.logspace
op_logspace <-
function (start, stop, num = 50L, endpoint = TRUE, base = 10L,
    dtype = NULL, axis = 1L)
{
    args <- capture_args(list(num = as_integer, base = as_integer,
        axis = as_axis))
    do.call(keras$ops$logspace, args)
}


#' Matrix product of two tensors.
#'
#' @description
#' - If both tensors are 1-dimensional, the dot product (scalar) is returned.
#' - If either tensor is N-D, N > 2, it is treated as a stack of matrices
#'   residing in the last two indexes and broadcast accordingly.
#' - If the first tensor is 1-D, it is promoted to a matrix by prepending
#'   a 1 to its dimensions. After matrix multiplication the prepended
#'   1 is removed.
#' - If the second tensor is 1-D, it is promoted to a matrix by appending a 1
#'   to its dimensions. After matrix multiplication the appended 1 is removed.
#'
#' @returns
#' Output tensor, matrix product of the inputs.
#'
#' @param x1
#' First tensor.
#'
#' @param x2
#' Second tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#matmul-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/matmul>
#' @tether keras.ops.matmul
op_matmul <-
function (x1, x2)
keras$ops$matmul(x1, x2)


#' Return the maximum of a tensor or maximum along an axis.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' (x <- op_convert_to_tensor(rbind(c(1, 3, 5), c(1, 5, 2))))
#' op_max(x)
#' op_max(x, axis = 1)
#' op_max(x, axis = 1, keepdims = TRUE)
#' ```
#'
#' @returns
#' Maximum of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which to operate. By default, flattened input
#' is used.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduced are left
#' in the result as dimensions with size one. Defaults to`FALSE`.
#'
#' @param initial
#' The minimum value of an output element. Defaults to`NULL`.
#'
#' @export
#' @aliases op_amax
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#max-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/max>
#' @tether keras.ops.max
op_max <-
function (x, axis = NULL, keepdims = FALSE, initial = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$max, args)
}


#' Element-wise maximum of `x1` and `x2`.
#'
#' @returns
#' Output tensor, element-wise maximum of `x1` and `x2`.
#'
#' @param x1
#' First tensor.
#'
#' @param x2
#' Second tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#maximum-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/maximum>
#' @tether keras.ops.maximum
op_maximum <-
function (x1, x2)
keras$ops$maximum(x1, x2)

#' @export
#' @rdname op_maximum
op_pmax <- op_maximum


#' Compute the arithmetic mean along the specified axes.
#'
#' @returns
#' Output tensor containing the mean values.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which the means are computed. The default
#' is to compute the mean of the flattened tensor.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduced are left
#' in the result as dimensions with size one.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#mean-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/mean>
#' @tether keras.ops.mean
op_mean <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$mean, args)
}


#' Compute the median along the specified axis.
#'
#' @returns
#' The output tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which the medians are computed. Defaults to
#' `axis = NULL` which is to compute the median(s) along a flattened
#' version of the array.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduce
#' are left in the result as dimensions with size one.
#'
#' @export
#' @family numpy ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/median>
#' @tether keras.ops.median
op_median <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$median, args)
}


#' Creates grids of coordinates from coordinate vectors.
#'
#' @description
#' Given `N` 1-D tensors `T0, T1, ..., TN-1` as inputs with corresponding
#' lengths `S0, S1, ..., SN-1`, this creates an `N` N-dimensional tensors
#' `G0, G1, ..., GN-1` each with shape `(S0, ..., SN-1)` where the output
#' `Gi` is constructed by expanding `Ti` to the result shape.
#'
#' # Examples
#' ```{r}
#' x <- op_array(c(1, 2, 3), "int32")
#' y <- op_array(c(4, 5, 6), "int32")
#' ```
#'
#' ```{r}
#' c(grid_x, grid_y) %<-% op_meshgrid(x, y, indexing = "ij")
#' grid_x
#' # array([[1, 1, 1],
#' #        [2, 2, 2],
#' #        [3, 3, 3]))
#' grid_y
#' # array([[4, 5, 6],
#' #        [4, 5, 6],
#' #        [4, 5, 6]))
#' ```
#'
#' @returns
#' Sequence of N tensors.
#'
#' @param ...
#' 1-D tensors representing the coordinates of a grid.
#'
#' @param indexing
#' `"xy"` or `"ij"`. "xy" is cartesian; `"ij"` is matrix
#' indexing of output. Defaults to `"xy"`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#meshgrid-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/meshgrid>
#'
#' @tether keras.ops.meshgrid
op_meshgrid <-
function (..., indexing = "xy")
{
    args <- lapply(list(...), function(x) {
        if (storage.mode(x) == "double")
            np_array(x, "int64")
        else x
    })
    keras$ops$meshgrid(!!!args, indexing = indexing)
}


#' Return the minimum of a tensor or minimum along an axis.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' (x <- op_convert_to_tensor(rbind(c(1, 3, 5), c(1, 5, 2))))
#' op_min(x)
#' op_min(x, axis = 1)
#' op_min(x, axis = 1, keepdims = TRUE)
#' ```
#' @returns
#' Minimum of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which to operate. By default, flattened input
#' is used.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduced are left
#' in the result as dimensions with size one. Defaults to`FALSE`.
#'
#' @param initial
#' The maximum value of an output element. Defaults to`NULL`.
#'
#' @export
#' @aliases op_amin
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#min-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/min>
#' @tether keras.ops.min
op_min <-
function (x, axis = NULL, keepdims = FALSE, initial = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$min, args)
}


#' Element-wise minimum of `x1` and `x2`.
#'
#' @returns
#' Output tensor, element-wise minimum of `x1` and `x2`.
#'
#' @param x1
#' First tensor.
#'
#' @param x2
#' Second tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#minimum-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/minimum>
#' @tether keras.ops.minimum
op_minimum <-
function (x1, x2)
keras$ops$minimum(x1, x2)

#' @rdname op_minimum
#' @export
op_pmin <- op_minimum


#' Returns the element-wise remainder of division.
#'
#' @returns
#' Output tensor, element-wise remainder of division.
#'
#' @param x1
#' First tensor.
#'
#' @param x2
#' Second tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#mod-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/mod>
#' @tether keras.ops.mod
op_mod <-
function (x1, x2)
keras$ops$mod(x1, x2)


#' Move axes of a tensor to new positions.
#'
#' @description
#' Other axes remain in their original order.
#'
#' @returns
#' Tensor with moved axes.
#'
#' @param x
#' Tensor whose axes should be reordered.
#'
#' @param source
#' Original positions of the axes to move. These must be unique.
#'
#' @param destination
#' Destinations positions for each of the original axes.
#' These must also be unique.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#moveaxis-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/moveaxis>
#' @tether keras.ops.moveaxis
op_moveaxis <-
function (x, source, destination)
keras$ops$moveaxis(x, as_axis(source), as_axis(destination))


#' Multiply arguments element-wise.
#'
#' @returns
#' Output tensor, element-wise product of `x1` and `x2`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#multiply-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/multiply>
#' @tether keras.ops.multiply
op_multiply <-
function (x1, x2)
keras$ops$multiply(x1, x2)


#' Replace NaN with zero and infinity with large finite numbers.
#'
#' @returns
#' `x`, with non-finite values replaced.
#'
#' @param x
#' Input data.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#nantonum-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/nan_to_num>
#' @tether keras.ops.nan_to_num
op_nan_to_num <-
function (x)
keras$ops$nan_to_num(x)


#' Return the number of dimensions of a tensor.
#'
#' @returns
#' The number of dimensions in `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#ndim-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/ndim>
#' @tether keras.ops.ndim
op_ndim <-
function (x)
keras$ops$ndim(x)


#' Numerical negative, element-wise.
#'
#' @returns
#' Output tensor, `y = -x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#negative-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/negative>
#' @tether keras.ops.negative
op_negative <-
function (x)
keras$ops$negative(x)


#' Return the indices of the elements that are non-zero.
#'
#' @returns
#' Indices of elements that are non-zero.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#nonzero-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/nonzero>
#' @tether keras.ops.nonzero
op_nonzero <-
function (x)
keras$ops$nonzero(x)


#' Return `(x1 != x2)` element-wise.
#'
#' @returns
#' Output tensor, element-wise comparsion of `x1` and `x2`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#notequal-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/not_equal>
#' @tether keras.ops.not_equal
op_not_equal <-
function (x1, x2)
keras$ops$not_equal(x1, x2)


#' Return a new tensor of given shape and type, filled with ones.
#'
#' @returns
#' Tensor of ones with the given shape and dtype.
#'
#' @param shape
#' Shape of the new tensor.
#'
#' @param dtype
#' Desired data type of the tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#ones-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/ones>
#' @tether keras.ops.ones
op_ones <-
function (shape, dtype = NULL)
{
    args <- capture_args(list(shape = normalize_shape))
    do.call(keras$ops$ones, args)
}


#' Return a tensor of ones with the same shape and type of `x`.
#'
#' @returns
#' A tensor of ones with the same shape and type as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param dtype
#' Overrides the data type of the result.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#oneslike-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/ones_like>
#' @tether keras.ops.ones_like
op_ones_like <-
function (x, dtype = NULL)
keras$ops$ones_like(x, dtype)


#' Compute the outer product of two vectors.
#'
#' @description
#' Given two vectors `x1` and `x2`, the outer product is:
#'
#' ```
#' out[i, j] = x1[i] * x2[j]
#' ```
#'
#' @returns
#' Outer product of `x1` and `x2`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#outer-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/outer>
#' @tether keras.ops.outer
op_outer <-
function (x1, x2)
keras$ops$outer(x1, x2)


#' Pad a tensor.
#'
#' @description
#'
#' # Note
#' Torch backend only supports modes `"constant"`, `"reflect"`,
#'     `"symmetric"` and `"circular"`.
#'     Only Torch backend supports `"circular"` mode.
#'
#' Note:
#'     Tensorflow backend only supports modes `"constant"`, `"reflect"`
#'     and `"symmetric"`.
#'
#' @returns
#' Padded tensor.
#'
#' @param x
#' Tensor to pad.
#'
#' @param pad_width
#' Number of values padded to the edges of each axis.
#' `((before_1, after_1), ...(before_N, after_N))` unique pad
#' widths for each axis.
#' `((before, after),)` yields same before and after pad for
#' each axis.
#' `(pad,)` or `int` is a shortcut for `before = after = pad`
#' width for all axes.
#'
#' @param mode
#' One of `"constant"`, `"edge"`, `"linear_ramp"`,
#' `"maximum"`, `"mean"`, `"median"`, `"minimum"`,
#' `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`,
#' `"circular"`. Defaults to`"constant"`.
#'
#' @param constant_values
#' Value to pad with if `mode == "constant"`.
#' Defaults to `0`. A `ValueError` is raised if not `NULL` and
#' `mode != "constant"`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#pad-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/pad>
#' @tether keras.ops.pad
op_pad <-
function (x, pad_width, mode = "constant", constant_values = NULL)
{
    args <- capture_args(list(pad_width = as_integer))
    do.call(keras$ops$pad, args)
}


#' First tensor elements raised to powers from second tensor, element-wise.
#'
#' @returns
#' Output tensor, the bases in `x1` raised to the exponents in `x2`.
#'
#' @param x1
#' The bases.
#'
#' @param x2
#' The exponents.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#power-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/power>
#' @tether keras.ops.power
op_power <-
function (x1, x2)
keras$ops$power(x1, x2)


#' Return the product of tensor elements over a given axis.
#'
#' @returns
#' Product of elements of `x` over the given axis or axes.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which a product is performed. The default,
#' `axis = NULL`, will compute the product of all elements
#' in the input tensor.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduce
#' are left in the result as dimensions with size one.
#'
#' @param dtype
#' Data type of the returned tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#prod-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/prod>
#' @tether keras.ops.prod
op_prod <-
function (x, axis = NULL, keepdims = FALSE, dtype = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$prod, args)
}


#' Compute the q-th quantile(s) of the data along the specified axis.
#'
#' @returns
#' The quantile(s). If `q` is a single probability and `axis=NULL`, then
#' the result is a scalar. If multiple probabilies levels are given, first
#' axis of the result corresponds to the quantiles. The other axes are the
#' axes that remain after the reduction of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param q
#' Probability or sequence of probabilities for the quantiles to
#' compute. Values must be between 0 and 1 inclusive.
#'
#' @param axis
#' Axis or axes along which the quantiles are computed. Defaults to
#' `axis=NULL` which is to compute the quantile(s) along a flattened
#' version of the array.
#'
#' @param method
#' A string specifies the method to use for estimating the
#' quantile. Available methods are `"linear"`, `"lower"`, `"higher"`,
#' `"midpoint"`, and `"nearest"`. Defaults to `"linear"`.
#' If the desired quantile lies between two data points `i < j`:
#' - `"linear"`: `i + (j - i) * fraction`, where fraction is the
#'     fractional part of the index surrounded by `i` and `j`.
#' - `"lower"`: `i`.
#' - `"higher"`: `j`.
#' - `"midpoint"`: `(i + j) / 2`
#' - `"nearest"`: `i` or `j`, whichever is nearest.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduce
#' are left in the result as dimensions with size one.
#'
#' @export
#' @family numpy ops
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/quantile>
#' @tether keras.ops.quantile
op_quantile <-
function (x, q, axis = NULL, method = "linear", keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$quantile, args)
}


#' Return a contiguous flattened tensor.
#'
#' @description
#' A 1-D tensor, containing the elements of the input, is returned.
#'
#' @returns
#' Output tensor.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#ravel-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/ravel>
#' @tether keras.ops.ravel
op_ravel <-
function (x)
keras$ops$ravel(x)


#' Return the real part of the complex argument.
#'
#' @returns
#' The real component of the complex argument.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#real-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/real>
#' @tether keras.ops.real
op_real <-
function (x)
keras$ops$real(x)


#' Return the reciprocal of the argument, element-wise.
#'
#' @description
#' Calculates `1/x`.
#'
#' @returns
#' Output tensor, element-wise reciprocal of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#reciprocal-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/reciprocal>
#' @tether keras.ops.reciprocal
op_reciprocal <-
function (x)
keras$ops$reciprocal(x)


#' Repeat each element of a tensor after themselves.
#'
#' @returns
#' Output tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param repeats
#' The number of repetitions for each element.
#'
#' @param axis
#' The axis along which to repeat values. By default, use
#' the flattened input array, and return a flat output array.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#repeat-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/repeat>
#' @tether keras.ops.repeat
op_repeat <-
function (x, repeats, axis = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$`repeat`, args)
}


#' Gives a new shape to a tensor without changing its data.
#'
#' @returns
#' The reshaped tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param newshape
#' The new shape should be compatible with the original shape.
#' One shape dimension can be `-1` in which case the value is
#' inferred from the length of the array and remaining dimensions.
#'
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#reshape-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/reshape>
#' @tether keras.ops.reshape
op_reshape <-
function (x, newshape)
{
    keras$ops$reshape(x, tuple(lapply(shape(newshape),
                                      function(d) d %||% -1L)))
}


#' Roll tensor elements along a given axis.
#'
#' @description
#' Elements that roll beyond the last position are re-introduced at the first.
#'
#' @returns
#' Output tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param shift
#' The number of places by which elements are shifted.
#'
#' @param axis
#' The axis along which elements are shifted. By default, the
#' array is flattened before shifting, after which the original
#' shape is restored.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#roll-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/roll>
#' @tether keras.ops.roll
op_roll <-
function (x, shift, axis = NULL)
{
    args <- capture_args(list(shift = as_integer, axis = as_axis))
    do.call(keras$ops$roll, args)
}


#' Evenly round to the given number of decimals.
#'
#' @returns
#' Output tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param decimals
#' Number of decimal places to round to. Defaults to `0`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#round-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/round>
#' @tether keras.ops.round
op_round <-
function (x, decimals = 0L)
{
    args <- capture_args(list(decimals = as_integer))
    do.call(keras$ops$round, args)
}


#' Returns a tensor with the signs of the elements of `x`.
#'
#' @returns
#' Output tensor of same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#sign-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sign>
#' @tether keras.ops.sign
op_sign <-
function (x)
keras$ops$sign(x)


#' Trigonomeric sine, element-wise.
#'
#' @returns
#' Output tensor of same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#sin-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sin>
#' @tether keras.ops.sin
op_sin <-
function (x)
keras$ops$sin(x)


#' Hyperbolic sine, element-wise.
#'
#' @returns
#' Output tensor of same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#sinh-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sinh>
#' @tether keras.ops.sinh
op_sinh <-
function (x)
keras$ops$sinh(x)


#' Return the number of elements in a tensor.
#'
#' @returns
#' Number of elements in `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#size-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/size>
#' @tether keras.ops.size
op_size <-
function (x)
keras$ops$size(x)


#' Sorts the elements of `x` along a given axis in ascending order.
#'
#' @returns
#' Sorted tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis along which to sort. If `NULL`, the tensor is flattened
#' before sorting. Defaults to `-1`; the last axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#sort-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sort>
#' @tether keras.ops.sort
op_sort <-
function (x, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$sort, args)
}


#' Split a tensor into chunks.
#'
#' @description
#'
#' # Note
#' A split does not have to result in equal division when using
#' Torch backend.
#'
#' @returns
#' A list of tensors.
#'
#' @param x
#' Input tensor.
#'
#' @param indices_or_sections
#' If an integer, N, the tensor will be split into N
#' equal sections along `axis`. If a 1-D array of sorted integers,
#' the entries indicate indices at which the tensor will be split
#' along `axis`.
#'
#' @param axis
#' Axis along which to split. Defaults to `1`, the first axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#split-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/split>
#' @tether keras.ops.split
op_split <-
function (x, indices_or_sections, axis = 1L)
{
    args <- capture_args(list(indices_or_sections = as_integer,
        axis = as_axis))
    do.call(keras$ops$split, args)
}


#' Return the non-negative square root of a tensor, element-wise.
#'
#' @returns
#' Output tensor, the non-negative square root of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#sqrt-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sqrt>
#' @tether keras.ops.sqrt
op_sqrt <-
function (x)
keras$ops$sqrt(x)


#' Return the element-wise square of the input.
#'
#' @returns
#' Output tensor, the square of `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#square-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/square>
#' @tether keras.ops.square
op_square <-
function (x)
keras$ops$square(x)


#' Remove axes of length one from `x`.
#'
#' @returns
#' The input tensor with all or a subset of the dimensions of
#' length 1 removed.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Select a subset of the entries of length one in the shape.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#squeeze-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/squeeze>
#' @tether keras.ops.squeeze
op_squeeze <-
function (x, axis = NULL)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$squeeze, args)
}


#' Join a sequence of tensors along a new axis.
#'
#' @description
#' The `axis` parameter specifies the index of the new axis in the
#' dimensions of the result.
#'
#' @returns
#' The stacked tensor.
#'
#' @param x
#' A sequence of tensors.
#'
#' @param axis
#' Axis along which to stack. Defaults to `1`, the first axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#stack-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/stack>
#' @tether keras.ops.stack
op_stack <-
function (x, axis = 1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$stack, args)
}


#' Compute the standard deviation along the specified axis.
#'
#' @returns
#' Output tensor containing the standard deviation values.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis along which to compute standard deviation.
#' Default is to compute the standard deviation of the
#' flattened tensor.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduced are left
#' in the result as dimensions with size one.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#std-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/std>
#' @tether keras.ops.std
op_std <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$std, args)
}


#' Subtract arguments element-wise.
#'
#' @returns
#' Output tensor, element-wise difference of `x1` and `x2`.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#subtract-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/subtract>
#' @tether keras.ops.subtract
op_subtract <-
function (x1, x2)
keras$ops$subtract(x1, x2)


#' Sum of a tensor over the given axes.
#'
#' @returns
#' Output tensor containing the sum.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which the sum is computed. The default is to
#' compute the sum of the flattened tensor.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduced are left
#' in the result as dimensions with size one.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#sum-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sum>
#' @tether keras.ops.sum
op_sum <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$sum, args)
}


#' Interchange two axes of a tensor.
#'
#' @returns
#' A tensor with the axes swapped.
#'
#' @param x
#' Input tensor.
#'
#' @param axis1
#' First axis.
#'
#' @param axis2
#' Second axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#swapaxes-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/swapaxes>
#' @tether keras.ops.swapaxes
op_swapaxes <-
function (x, axis1, axis2)
keras$ops$swapaxes(x, axis1, axis2)


#' Take elements from a tensor along an axis.
#'
#' @returns
#' The corresponding tensor of values.
#'
#' @param x
#' Source tensor.
#'
#' @param indices
#' The indices of the values to extract.
#'
#' @param axis
#' The axis over which to select values. By default, the
#' flattened input tensor is used.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#take-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/take>
#' @tether keras.ops.take
op_take <-
function (x, indices, axis = NULL)
{
    args <- capture_args(list(indices = as_index, axis = as_axis))
    do.call(keras$ops$take, args)
}


#' Select values from `x` at the 1-D `indices` along the given axis.
#'
#' @returns
#' The corresponding tensor of values.
#'
#' @param x
#' Source tensor.
#'
#' @param indices
#' The indices of the values to extract.
#'
#' @param axis
#' The axis over which to select values. By default, the flattened
#' input tensor is used.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#takealongaxis-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/take_along_axis>
#' @tether keras.ops.take_along_axis
op_take_along_axis <-
function (x, indices, axis = NULL)
{
    args <- capture_args(list(indices = as_index, axis = as_axis))
    do.call(keras$ops$take_along_axis, args)
}


#' Compute tangent, element-wise.
#'
#' @returns
#' Output tensor of same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#tan-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/tan>
#' @tether keras.ops.tan
op_tan <-
function (x)
keras$ops$tan(x)


#' Hyperbolic tangent, element-wise.
#'
#' @returns
#' Output tensor of same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/nn#tanh-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/tanh>
#' @tether keras.ops.tanh
op_tanh <-
function (x)
keras$ops$tanh(x)


#' Compute the tensor dot product along specified axes.
#'
#' @returns
#' The tensor dot product of the inputs.
#'
#' @param x1
#' First tensor.
#'
#' @param x2
#' Second tensor.
#'
#' @param axes
#' - If an integer, N, sum over the last N axes of `x1` and the
#'   first N axes of `x2` in order. The sizes of the corresponding
#'   axes must match.
#' - Or, a list of axes to be summed over, first sequence applying
#'   to `x1`, second to `x2`. Both sequences must be of the
#'   same length.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#tensordot-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/tensordot>
#' @tether keras.ops.tensordot
op_tensordot <-
function (x1, x2, axes = 3L)
{
    args <- capture_args(list(axes = as_axis))
    do.call(keras$ops$tensordot, args)
}


#' Repeat `x` the number of times given by `repeats`.
#'
#' @description
#' If `repeats` has length `d`, the result will have dimension of
#' `max(d, x.ndim)`.
#'
#' If `x.ndim < d`, `x` is promoted to be d-dimensional by prepending
#' new axes.
#'
#' If `x.ndim > d`, `repeats` is promoted to `x.ndim` by prepending 1's to it.
#'
#' @returns
#' The tiled output tensor.
#'
#' @param x
#' Input tensor.
#'
#' @param repeats
#' The number of repetitions of `x` along each axis.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#tile-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/tile>
#' @tether keras.ops.tile
op_tile <-
function (x, repeats)
keras$ops$tile(x, repeats)


#' Return the sum along diagonals of the tensor.
#'
#' @description
#' If `x` is 2-D, the sum along its diagonal with the given offset is
#' returned, i.e., the sum of elements `x[i, i+offset]` for all `i`.
#'
#' If a has more than two dimensions, then the axes specified by `axis1`
#' and `axis2` are used to determine the 2-D sub-arrays whose traces are
#' returned.
#'
#' The shape of the resulting tensor is the same as that of `x` with `axis1`
#' and `axis2` removed.
#'
#' @returns
#' If `x` is 2-D, the sum of the diagonal is returned. If `x` has
#' larger dimensions, then a tensor of sums along diagonals is
#' returned.
#'
#' @param x
#' Input tensor.
#'
#' @param offset
#' Offset of the diagonal from the main diagonal. Can be
#' both positive and negative. Defaults to `0`.
#'
#' @param axis1
#' Axis to be used as the first axis of the 2-D sub-arrays.
#' Defaults to `1`. (first axis).
#'
#' @param axis2
#' Axis to be used as the second axis of the 2-D sub-arrays.
#' Defaults to `2`. (second axis).
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#trace-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/trace>
#' @tether keras.ops.trace
op_trace <-
function (x, offset = 0L, axis1 = 1L, axis2 = 2L)
{
    args <- capture_args(list(offset = as_integer, axis1 = as_integer,
        axis2 = as_integer))
    do.call(keras$ops$trace, args)
}


#' Returns a tensor with `axes` transposed.
#'
#' @returns
#' `x` with its axes permuted.
#'
#' @param x
#' Input tensor.
#'
#' @param axes
#' Sequence of integers. Permutation of the dimensions of `x`.
#' By default, the order of the axes are reversed.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#transpose-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/transpose>
#' @tether keras.ops.transpose
op_transpose <-
function (x, axes = NULL)
{
    args <- capture_args(list(axes = as_axis))
    do.call(keras$ops$transpose, args)
}


#' Return a tensor with ones at and below a diagonal and zeros elsewhere.
#'
#' @returns
#' Tensor with its lower triangle filled with ones and zeros elsewhere.
#' `T[i, j] == 1` for `j <= i + k`, 0 otherwise.
#'
#' @param N
#' Number of rows in the tensor.
#'
#' @param M
#' Number of columns in the tensor.
#'
#' @param k
#' The sub-diagonal at and below which the array is filled.
#' `k = 0` is the main diagonal, while `k < 0` is below it, and
#' `k > 0` is above. The default is 0.
#'
#' @param dtype
#' Data type of the returned tensor. The default is "float32".
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#tri-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/tri>
#' @tether keras.ops.tri
op_tri <-
function (N, M = NULL, k = 0L, dtype = NULL)
{
    args <- capture_args(list(k = as_integer))
    do.call(keras$ops$tri, args)
}


#' Return lower triangle of a tensor.
#'
#' @description
#' For tensors with `ndim` exceeding 2, `tril` will apply to the
#' final two axes.
#'
#' @returns
#' Lower triangle of `x`, of same shape and data type as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param k
#' Diagonal above which to zero elements. Defaults to `0`. the
#' main diagonal. `k < 0` is below it, and `k > 0` is above it.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#tril-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/tril>
#' @tether keras.ops.tril
op_tril <-
function (x, k = 0L)
{
    args <- capture_args(list(k = as_integer))
    do.call(keras$ops$tril, args)
}


#' Return upper triangle of a tensor.
#'
#' @description
#' For tensors with `ndim` exceeding 2, `triu` will apply to the
#' final two axes.
#'
#' @returns
#' Upper triangle of `x`, of same shape and data type as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param k
#' Diagonal below which to zero elements. Defaults to `0`. the
#' main diagonal. `k < 0` is below it, and `k > 0` is above it.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#triu-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/triu>
#' @tether keras.ops.triu
op_triu <-
function (x, k = 0L)
{
    args <- capture_args(list(k = as_integer))
    do.call(keras$ops$triu, args)
}


# ' Alias for `keras.ops.divide`.
# '
# ' @param x1
# ' see description
# '
# ' @param x2
# ' see description
# '
# ' @export
# ' @family numpy ops
# ' @family ops
# ' @seealso
# ' + <https://keras.io/api/ops/numpy#truedivide-function>
# ' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/true_divide>
# ' @tether keras.ops.true_divide
# op_true_divide <-
# function (x1, x2)
# keras$ops$true_divide(x1, x2)


#' Compute the variance along the specified axes.
#'
#' @returns
#' Output tensor containing the variance.
#'
#' @param x
#' Input tensor.
#'
#' @param axis
#' Axis or axes along which the variance is computed. The default
#' is to compute the variance of the flattened tensor.
#'
#' @param keepdims
#' If this is set to `TRUE`, the axes which are reduced are left
#' in the result as dimensions with size one.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#var-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/var>
#' @tether keras.ops.var
op_var <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$var, args)
}


#' Return the dot product of two vectors.
#'
#' @description
#' If the first argument is complex, the complex conjugate of the first
#' argument is used for the calculation of the dot product.
#'
#' Multidimensional tensors are flattened before the dot product is taken.
#'
#' @returns
#' Output tensor.
#'
#' @param x1
#' First input tensor. If complex, its complex conjugate is taken
#' before calculation of the dot product.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#vdot-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/vdot>
#' @tether keras.ops.vdot
op_vdot <-
function (x1, x2)
keras$ops$vdot(x1, x2)


#' Stack tensors in sequence vertically (row wise).
#'
#' @returns
#' Tensor formed by stacking the given tensors.
#'
#' @param xs
#' Sequence of tensors.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#vstack-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/vstack>
#' @tether keras.ops.vstack
op_vstack <-
function (xs)
keras$ops$vstack(xs)


#' Return elements chosen from `x1` or `x2` depending on `condition`.
#'
#' @returns
#' A tensor with elements from `x1` where `condition` is `TRUE`, and
#' elements from `x2` where `condition` is `FALSE`.
#'
#' @param condition
#' Where `TRUE`, yield `x1`, otherwise yield `x2`.
#'
#' @param x1
#' Values from which to choose when `condition` is `TRUE`.
#'
#' @param x2
#' Values from which to choose when `condition` is `FALSE`.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#where-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/where>
#' @tether keras.ops.where
op_where <-
function (condition, x1 = NULL, x2 = NULL)
keras$ops$where(condition, x1, x2)


#' Return a new tensor of given shape and type, filled with zeros.
#'
#' @returns
#' Tensor of zeros with the given shape and dtype.
#'
#' @param shape
#' Shape of the new tensor.
#'
#' @param dtype
#' Desired data type of the tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#zeros-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/zeros>
#' @tether keras.ops.zeros
op_zeros <-
function (shape, dtype = NULL)
{
    args <- capture_args(list(shape = normalize_shape))
    do.call(keras$ops$zeros, args)
}


#' Return a tensor of zeros with the same shape and type as `x`.
#'
#' @returns
#' A tensor of zeros with the same shape and type as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @param dtype
#' Overrides the data type of the result.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/numpy#zeroslike-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/zeros_like>
#' @tether keras.ops.zeros_like
op_zeros_like <-
function (x, dtype = NULL)
keras$ops$zeros_like(x, dtype)


#' CTC (Connectionist Temporal Classification) loss.
#'
#' @param target
#' A tensor of shape `(batch_size, max_length)` containing
#' the true labels in integer format.
#'
#' @param output
#' A tensor of shape `(batch_size, max_length, num_classes)`
#' containing logits (the output of your model).
#'
#' @param target_length
#' A tensor of shape `(batch_size)` containing the
#' true label lengths.
#'
#' @param output_length
#' A tensor of shape `(batch_size)` containing the
#' output lengths.
#'
#' @param mask_index
#' The index of the mask character in the vocabulary.
#' Defaults to `0`.
#'
#' @returns A tensor, shape `(batch_size)`, of loss values.
#' @export
#' @family nn ops
#' @family ops
#' @tether keras.ops.ctc_loss
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/ctc_loss>
op_ctc_loss <-
function (target, output, target_length, output_length, mask_index = 0L)
{
    args <- capture_args(list(target = as_integer_array, mask_index = as_integer))
    do.call(keras$ops$ctc_loss, args)
}


#' Hard SiLU activation function, also known as Hard Swish.
#'
#' @description
#' It is defined as:
#'
#' - `0` if `if x < -3`
#' - `x` if `x > 3`
#' - `x * (x + 3) / 6` if `-3 <= x <= 3`
#'
#' It's a faster, piecewise linear approximation of the silu activation.
#'
#' # Examples
#' ```{r}
#' x <- op_convert_to_tensor(c(-3.0, -1.0, 0.0, 1.0, 3.0))
#' op_hard_silu(x)
#' ```
#'
#' @returns
#' A tensor with the same shape as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @family nn ops
#' @family ops
#' @tether keras.ops.hard_silu
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/hard_silu>
op_hard_silu <-
function (x)
keras$ops$hard_silu(x)

#' @rdname op_hard_silu
#' @export
op_hard_swish <-
function (x)
keras$ops$hard_swish(x)
