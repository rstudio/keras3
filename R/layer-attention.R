
#' Creates attention layer
#' 
#' Dot-product attention layer, a.k.a. Luong-style attention.
#' 
#' @inheritParams layer_dense
#' 
#' @param inputs a list of inputs first should be the query tensor, the second the value tensor
#' @param use_scale If True, will create a scalar variable to scale the attention scores.
#' @param causal Boolean. Set to True for decoder self-attention. Adds a mask such that position i cannot attend to positions j > i. 
#' This prevents the flow of information from the future towards the past.
#'
#' @family core layers
#' @family attention layers   
#'         
#' @export
layer_attention <- function(inputs,use_scale=FALSE, causal = FALSE, batch_size = NULL, dtype = NULL, 
                                name = NULL, trainable = NULL, weights = NULL) {
  if (!is_tensorflow_implementation() || !tensorflow::tf_version() >= "1.14")
    stop("layer_dense_features requires TensorFlow implementation and version >= 1.14")
  create_layer(keras$layers$Attention, inputs, list(
      use_scale = use_scale,
      causal = causal,
      batch_size = batch_size,
      dtype = dtype, 
      name = name,
      trainable = trainable, 
      weights = weights)
    )
    
    
}



