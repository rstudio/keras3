

#' This is the class from which all layers inherit
#'
#' @details
#'
#' A layer is a callable object that takes as input one or more tensors and
#' that outputs one or more tensors. It involves *computation*, defined
#' in the `call()` method, and a *state* (weight variables), defined
#' either in the constructor `__init__()` or in the `build()` method.
#'
#' Users will just instantiate a layer and then treat it as a callable.
#'
#' @param trainable Boolean, whether the layer's variables should be trainable.
#'
#' @param dynamic Set this to `TRUE` if your layer should only be run eagerly, and
#' should not be used to generate a static computation graph.
#' This would be the case for a Tree-RNN or a recursive network,
#' for example, or generally for any layer that manipulates tensors
#' using Python control flow. If `FALSE`, we assume that the layer can
#' safely be used to generate a static computation graph.
#'
#'
#' @attribute variable_dtype Alias of `dtype`.
#'
#' @attribute compute_dtype The dtype of the layer's computations. Layers automatically
#' cast inputs to this dtype which causes the computations and output to also
#' be in this dtype. When mixed precision is used with a
#' `tf.keras.mixed_precision.Policy`, this will be different than
#' `variable_dtype`.
#'
#' @attribute dtype_policy The layer's dtype policy. See the
#' `tf.keras.mixed_precision.Policy` documentation for details.
#'
#' @attribute trainable_weights List of variables to be included in backprop.
#'
#' @attribute non_trainable_weights List of variables that should not be
#' included in backprop.
#'
#' @attribute weights The concatenation of the lists trainable_weights and
#' non_trainable_weights (in this order).
#'
#' @attribute trainable Whether the layer should be trained (boolean), i.e. whether
#' its potentially-trainable weights should be returned as part of
#' `layer.trainable_weights`.
#'
#' @attribute input_spec Optional (list of) `InputSpec` object(s) specifying the
#' constraints on inputs that can be accepted by the layer.
#'
tf.keras.layers.Layer <-
  function(trainable = TRUE,
           name = NULL,
           dtype = NULL,
           dynamic = FALSE,
           ...
  )
stop("This function exists only for documentation purposes and is not intended to be called")
