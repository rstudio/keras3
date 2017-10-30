

#' Freeze and unfreeze layers
#' 
#' Freeze layers in a model so that they are no longer trainable.
#' 
#' @param object Keras model object
#' @param from Layer instance, layer name, or layer index within model
#' @param from Layer instance, layer name, or layer index within model 
#' 
#' @note Models must be compiled again after layers are frozen
#'  or unfrozen.
#' 
#' @examples \dontrun{
#' # instantiate a VGG16 model
#' conv_base <- application_vgg16(
#'   weights = "imagenet",
#'   include_top = FALSE,
#'   input_shape = c(150, 150, 3)
#' )
#' 
#' # freeze it's layers
#' freeze_layers(conv_base)
#' 
#' # create a composite model that includes the base + more layers
#' model <- keras_model_sequential() %>% 
#'   conv_base %>% 
#'   layer_flatten() %>% 
#'   layer_dense(units = 256, activation = "relu") %>% 
#'   layer_dense(units = 1, activation = "sigmoid")
#' 
#' # compile
#' model %>% compile(
#'   loss = "binary_crossentropy",
#'   optimizer = optimizer_rmsprop(lr = 2e-5),
#'   metrics = c("accuracy")
#' )
#' 
#' # unfreeze layers from "block5_conv1" on
#' unfreeze_layers(conv_base, from = "block5_conv1")
#' 
#' # compile again since we froze or unfroze layers
#' model %>% compile(
#'   loss = "binary_crossentropy",
#'   optimizer = optimizer_rmsprop(lr = 2e-5),
#'   metrics = c("accuracy")
#' )
#' 
#' }
#' 
#' @export
freeze_layers <- function(object, from = NULL, to = NULL) {
  
  # object is still globally trainable if either from or to is specified
  object$trainable <- !missing(from) || !missing(to)
  
  # apply trainlable 
  apply_trainable(object, from, to, FALSE)
    
  # return model invisibly (for chaining)
  invisible(object)
}


#' @rdname freeze_layers 
#' @export
unfreeze_layers <- function(object, from = NULL, to = NULL) {
  
  # object always trainable after unfreeze
  object$trainable <- TRUE
  
  # apply trainable
  apply_trainable(object, from, to, TRUE)
  
  # return model invisibly (for chaining)
  invisible(object)
}

apply_trainable <- function(object, from, to, trainable) {
  
  # first resolve from and to into layer names
  layers <- object$layers
  
  # NULL means beginning and end respectively
  if (is.null(from))
    from <- layers[[1]]$name
  if (is.null(to))
    to <- layers[[length(layers)]]$name
  
  # layer instances become layer names
  if (is_layer(from))
    from <- from$name
  if (is_layer(to))
    to <- to$name
  
  # layer indexes become layer names
  if (is.numeric(from))
    from <- layers[[from]]$name
  if (is.numeric(to))
    to <- layers[[to]]$name
  
  # apply trainable property
  set_trainable <- FALSE
  for (layer in layers) {
    
    # flag to begin applying property
    if (layer$name == from)
      set_trainable <- TRUE
    
    # apply property
    if (set_trainable)
      layer$trainable <- trainable
    
    # flat to stop applying property
    if (layer$name == to)
      set_trainable <- FALSE
  }
}

