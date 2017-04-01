
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
time_distributed <- function(x, layer, input_shape = NULL) {

  # if layer is missing 
  if (is_sequential_model(x))
    
  
  call_layer(tf$contrib$keras$python$keras$layers$TimeDistributed, x, list(
    layer = layer,
    input_shape = normalize_shape(input_shape)
  ))
  
}


#' Bidirectional wrapper for RNNs.
#' 
#' @param layer Recurrent instance.
#' @param merge_mode Mode by which outputs of the forward and backward RNNs will
#'   be combined. One of 'sum', 'mul', 'concat', 'ave', NULL. If NULL, the
#'   outputs will not be combined, they will be returned as a list.
#' @param weights weights
#'   
#' @family layer wrappers
#'   
#' @export
bidirectional <- function(x, layer, merge_mode = "concat", weights = NULL, input_shape = NULL) {
  
  call_layer(tf$contrib$keras$python$keras$layers$Bidirectional, x, list(
    layer = layer,
    merge_mode = merge_mode,
    weights = weights,
    input_shape = normalize_shape(input_shape)
  ))
  
}





