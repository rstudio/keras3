#' Hub Layer
#' 
#' Wraps a Hub module (or a similar callable) for TF2 as a Keras Layer.
#' 
#' This layer wraps a callable object for use as a Keras layer. The callable 
#' object can be passed directly, or be specified by a string with a handle 
#' that gets passed to `hub_load()`.
#' 
#' The callable object is expected to follow the conventions detailed below. 
#' (These are met by TF2-compatible modules loaded from TensorFlow Hub.)
#' 
#' The callable is invoked with a single positional argument set to one tensor or 
#' a list of tensors containing the inputs to the layer. If the callable accepts 
#' a training argument, a boolean is passed for it. It is `TRUE` if this layer 
#' is marked trainable and called for training.
#' 
#' If present, the following attributes of callable are understood to have special 
#' meanings: variables: a list of all tf.Variable objects that the callable depends on. 
#' trainable_variables: those elements of variables that are reported as trainable 
#' variables of this Keras Layer when the layer is trainable. regularization_losses: 
#' a list of callables to be added as losses of this Keras Layer when the layer is 
#' trainable. Each one must accept zero arguments and return a scalar tensor.
#'
#' @param handle a callable object (subject to the conventions above), or a string 
#'   for which `hub_load()` returns such a callable. A string is required to save 
#'   the Keras config of this Layer.
#' 
#' @param trainable Boolean controlling whether this layer is trainable.
#' 
#' @param arguments optionally, a list with additional keyword arguments passed to 
#'   the callable. These must be JSON-serializable to save the Keras config of 
#'   this layer.
#'
#' @export
layer_hub <- function(handle, trainable = FALSE, arguments = NULL, ...) {
  
  if (!reticulate::py_module_available("tensorflow_hub"))
    stop("You need to install the tensorflow-hub module.")
  
  hub <- reticulate::import("tensorflow_hub")
  
  hub$KerasLayer(handle, trainable, arguments, ...)
}


