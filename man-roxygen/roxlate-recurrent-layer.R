#' @md
#' 
#' @section Statefulness in RNNs:
#' 
#' You can set RNN layers to be 'stateful', which means that the states
#' computed for the samples in one batch will be reused as initial states
#' for the samples in the next batch. This assumes a one-to-one mapping
#' between samples in different successive batches.
#' 
#' To enable statefulness:
#'   - Specify `stateful=TRUE` in the layer constructor.
#'   - Specify a fixed batch size for your model. For sequential models,
#'     pass `batch_input_shape = c(...)` to the first layer in your model.
#'     For functional models with 1 or more Input layers, pass 
#'     `batch_shape = c(...)` to all the first layers in your model.
#'     This is the expected shape of your inputs *including the batch size*.
#'     It should be a vector of integers, e.g. `c(32, 10, 100)`.
#'   - Specify `shuffle = FALSE` when calling fit().
#' 
#' To reset the states of your model, call `reset_states()` on either
#' a specific layer, or on your entire model.
#' 
#' @section Initial State of RNNs:
#' 
#' You can specify the initial state of RNN layers symbolically by calling
#' them with the keyword argument `initial_state`. The value of
#' `initial_state` should be a tensor or list of tensors representing
#' the initial state of the RNN layer.
#' 
#' You can specify the initial state of RNN layers numerically by
#' calling `reset_states` with the keyword argument `states`. The value of
#' `states` should be a numpy array or list of numpy arrays representing
#' the initial state of the RNN layer.
#'
#' @family recurrent layers

