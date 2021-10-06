#' @md
#'
#' @section Input shapes:
#'
#' N-D tensor with shape `(batch_size, timesteps, ...)`,
#' or `(timesteps, batch_size, ...)` when `time_major = TRUE`.
#'
#' @section Output shape:
#'
#' - if `return_state`: a list of tensors. The first tensor is
#' the output. The remaining tensors are the last states,
#' each with shape `(batch_size, state_size)`, where `state_size`
#' could be a high dimension tensor shape.
#' - if `return_sequences`: N-D tensor with shape `[batch_size, timesteps,
#' output_size]`, where `output_size` could be a high dimension tensor shape, or
#' `[timesteps, batch_size, output_size]` when `time_major` is `TRUE`
#' - else, N-D tensor with shape `[batch_size, output_size]`, where
#' `output_size` could be a high dimension tensor shape.
#'
#' @section Masking:
#'
#' This layer supports masking for input data with a variable number of
#' timesteps. To introduce masks to your data, use
#' [`layer_embedding()`] with the `mask_zero` parameter set to `TRUE`.
#'
#' @section Statefulness in RNNs:
#'
#' You can set RNN layers to be 'stateful', which means that the states computed
#' for the samples in one batch will be reused as initial states for the samples
#' in the next batch. This assumes a one-to-one mapping between samples in
#' different successive batches.
#'
#' For intuition behind statefulness, there is a helpful blog post here:
#' <https://philipperemy.github.io/keras-stateful-lstm/>
#'
#'
#' To enable statefulness:
#'   - Specify `stateful = TRUE` in the layer constructor.
#'
#'   - Specify a fixed batch size for your model. For sequential models,
#'     pass `batch_input_shape = list(...)` to the first layer in your model.
#'     For functional models with 1 or more Input layers, pass
#'     `batch_shape = list(...)` to all the first layers in your model.
#'     This is the expected shape of your inputs *including the batch size*.
#'     It should be a list of integers, e.g. `list(32, 10, 100)`.
#'     For dimensions which can vary (are not known ahead of time),
#'     use `NULL` in place of an integer, e.g. `list(32, NULL, NULL)`.
#'
#'   - Specify `shuffle = FALSE` when calling `fit()`.
#'
#' To reset the states of your model, call `layer$reset_states()` on either
#' a specific layer, or on your entire model.
#'
#' @section Initial State of RNNs:
#'
#' You can specify the initial state of RNN layers symbolically by calling them
#' with the keyword argument `initial_state.` The value of initial_state should
#' be a tensor or list of tensors representing the initial state of the RNN
#' layer.
#'
#' You can specify the initial state of RNN layers numerically by calling
#' `reset_states` with the named argument `states.` The value of `states` should
#' be an array or list of arrays representing the initial state of the RNN
#' layer.
#'
#' @section Passing external constants to RNNs:
#'
#' You can pass "external" constants to the cell using the `constants` named
#' argument of `RNN$__call__` (as well as `RNN$call`) method. This requires that the
#' `cell$call` method accepts the same keyword argument `constants`. Such constants
#' can be used to condition the cell transformation on additional static inputs
#' (not changing over time), a.k.a. an attention mechanism.
#'
#'
#' @family recurrent layers
#'
#' @seealso
#'  + <https://www.tensorflow.org/guide/keras/rnn>

