
#' Apply a layer to every temporal slice of an input.
#'
#' The input should be at least 3D, and the dimension of index one will be
#' considered to be the temporal dimension.
#'
#' Consider a batch of 32 samples,  where each sample is a sequence of 10 vectors of 16 dimensions. The batch
#' input shape of the layer is then `(32, 10, 16)`, and the `input_shape`, not
#' including the samples dimension, is `(10, 16)`. You can then use
#' `time_distributed` to apply a `layer_dense` to each of the 10 timesteps,
#' independently.
#'
#' @inheritParams layer_dense
#'
#' @param layer A layer instance.
#'
#' @family layer wrappers
#'
#' @export
time_distributed <- function(object, layer, input_shape = NULL,
                             batch_input_shape = NULL, batch_size = NULL, dtype = NULL,
                             name = NULL, trainable = NULL, weights = NULL) {

  create_layer(keras$layers$TimeDistributed, object, list(
    layer = layer,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))

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
