#' @md
#' 
#' @section Input shapes:
#' 
#' 3D tensor with shape `(batch_size, timesteps, input_dim)`,
#' (Optional) 2D tensors with shape `(batch_size, output_dim)`.
#' 
#' @section Output shape:
#' 
#' - if `return_state`: a list of tensors. The first tensor is
#' the output. The remaining tensors are the last states,
#' each with shape `(batch_size, units)`.
#' - if `return_sequences`: 3D tensor with shape
#' `(batch_size, timesteps, units)`.
#' - else, 2D tensor with shape `(batch_size, units)`.
#' 
#' @section Masking:
#' 
#' This layer supports masking for input data with a variable number
#' of timesteps. To introduce masks to your data,
#' use an embedding layer with the `mask_zero` parameter
#' set to `TRUE`.
#' 
#' @section Statefulness in RNNs:
#' 
#' You can set RNN layers to be 'stateful', which means that the states
#' computed for the samples in one batch will be reused as initial states
#' for the samples in the next batch. This assumes a one-to-one mapping
#' between samples in different successive batches. For intuition behind
#' statefulness, there is a helpful blog post here: 
#' <https://philipperemy.github.io/keras-stateful-lstm/>
#' 
#' To enable statefulness:
#'   - Specify `stateful = TRUE` in the layer constructor.
#'   - Specify a fixed batch size for your model. For sequential models,
#'     pass `batch_input_shape = c(...)` to the first layer in your model.
#'     For functional models with 1 or more Input layers, pass 
#'     `batch_shape = c(...)` to all the first layers in your model.
#'     This is the expected shape of your inputs *including the batch size*.
#'     It should be a vector of integers, e.g. `c(32, 10, 100)`.
#'     For dimensions which can vary (are not known ahead of time),
#'     use `NULL` in place of an integer, e.g. `c(32, NULL, NULL)`.
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

