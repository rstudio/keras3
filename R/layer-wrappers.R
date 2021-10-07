#' This layer wrapper allows to apply a layer to every temporal slice of an input
#'
#' @details
#' Every input should be at least 3D, and the dimension of index one of the
#' first input will be considered to be the temporal dimension.
#'
#' Consider a batch of 32 video samples, where each sample is a 128x128 RGB image
#' with `channels_last` data format, across 10 timesteps.
#' The batch input shape is `(32, 10, 128, 128, 3)`.
#'
#' You can then use `TimeDistributed` to apply the same `Conv2D` layer to each
#' of the 10 timesteps, independently:
#'
#' ```R
#' input <- layer_input(c(10, 128, 128, 3))
#' conv_layer <- layer_conv_2d(filters = 64, kernel_size = c(3, 3))
#' output <- input %>% time_distributed(conv_layer)
#' output$shape # TensorShape([None, 10, 126, 126, 64])
#' ```
#'
#' Because `TimeDistributed` applies the same instance of `Conv2D` to each of the
#' timestamps, the same set of weights are used at each timestamp.
#'
#' @inheritParams layer_dense
#' @param layer a `tf.keras.layers.Layer` instance.
#' @param ... standard layer arguments.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed>
#'
#' @family layer wrappers
#' @export
time_distributed <-
function(object, layer, ...)
{
  args <-
    capture_args(
      match.call(),
      list(
        input_shape = normalize_shape,
        batch_input_shape = normalize_shape,
        batch_size = as_nullable_integer
      ),
      ignore = "object"
    )
    create_layer(keras$layers$TimeDistributed, object, args)
}


#' Bidirectional wrapper for RNNs
#'
#' @inheritParams layer_dense
#'
#' @param layer A `RNN` layer instance, such as `layer_lstm()` or
#'   `layer_gru()`. It could also be a `keras$layers$Layer` instance that
#'   meets the following criteria:
#'
#'   1. Be a sequence-processing layer (accepts 3D+ inputs).
#'
#'   2. Have a `go_backwards`, `return_sequences` and `return_state` attribute
#'   (with the same semantics as for the `RNN` class).
#'
#'   3. Have an `input_spec` attribute.
#'
#'   4. Implement serialization via `get_config()` and `from_config()`. Note
#'   that the recommended way to create new RNN layers is to write a custom RNN
#'   cell and use it with `layer_rnn()`, instead of subclassing
#'   `keras$layers$Layer` directly.
#'
#'   5. When `returns_sequences = TRUE`, the output of the masked timestep will
#'   be zero regardless of the layer's original `zero_output_for_mask` value.
#'
#' @param merge_mode Mode by which outputs of the forward and backward RNNs will
#'   be combined. One of `'sum'`, `'mul'`, `'concat'`, `'ave'`, `NULL`. If
#'   `NULL`, the outputs will not be combined, they will be returned as a list.
#'   Default value is `'concat'`.
#'
#' @param weights Split and propagated to the `initial_weights` attribute on the
#'   forward and backward layer.
#'
#' @param backward_layer Optional `keras.layers.RNN`, or `keras.layers.Layer`
#'   instance to be used to handle backwards input processing. If
#'   `backward_layer` is not provided, the layer instance passed as the `layer`
#'   argument will be used to generate the backward layer automatically. Note
#'   that the provided `backward_layer` layer should have properties matching
#'   those of the `layer` argument, in particular it should have the same values
#'   for `stateful`, `return_states`, `return_sequences`, etc. In addition,
#'   `backward_layer` and `layer` should have different `go_backwards` argument
#'   values. A `ValueError` will be raised if these requirements are not met.
#'
#' @param ... standard layer arguments.
#'
#' @family layer wrappers
#' @seealso
#'
#' - <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional>
#' - <https://keras.io/api/layers/recurrent_layers/bidirectional/>
#'
#' @export
bidirectional <-
function(object, layer, merge_mode = "concat",
         weights = NULL, backward_layer = NULL, ...)
{
  args <- capture_args(
    match.call(),
    modifiers = list(
      input_shape = normalize_shape,
      batch_input_shape = normalize_shape,
      batch_size = as_nullable_integer
    ),
    ignore = "object"
  )
  create_layer(keras$layers$Bidirectional, object, args)
}
